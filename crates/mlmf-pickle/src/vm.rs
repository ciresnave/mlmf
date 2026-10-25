// SPDX-License-Identifier: MIT OR Apache-2.0
//! A pickle reader that builds a description of the object and runs nothing.
//!
//! This is **not** a general pickle implementation. It understands the
//! opcodes a `torch.save` checkpoint uses (protocols 2 to 4) and records
//! what each would have built: a `REDUCE` is stored as "this callable applied
//! to these arguments", never called.
//!
//! # Why it is bounded
//!
//! A pickle is attacker-controlled. Objects live in an arena and the stack
//! and memo hold indices into it, so a memo `GET` costs one index and a
//! pickle that references one object a million times stays a million
//! indices rather than a million copies. Every length is checked against the
//! bytes that remain before anything is allocated; the number of opcodes and
//! of objects is capped.
//!
//! # The allowlist (spec section 9 AD-3)
//!
//! The security boundary is which callables a `REDUCE` or `NEWOBJ` may name:
//! [`ALLOWED_REDUCE_TARGETS`]. Nothing is executed whatever it says, but a
//! checkpoint that calls anything else is refused rather than described, and
//! a test pins the list's exact content so a new entry cannot arrive as an
//! unnoticed one-line diff.

use std::collections::HashMap;

use crate::error::PickleError;

/// The callables a `REDUCE` or `NEWOBJ` may name, as `(module, name)`.
///
/// **Growth is permitted; growth unnoticed is not.** The test
/// `the_allowlist_is_exactly_these_entries` fails when this changes.
pub const ALLOWED_REDUCE_TARGETS: [(&str, &str); 4] = [
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._tensor", "_rebuild_from_type_v2"),
];

const MAX_OPS: usize = 20_000_000;
const MAX_NODES: usize = 4_000_000;

/// Index of an object in the arena.
pub type Id = usize;

/// What an object is. Containers hold the indices of their members.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// `GLOBAL`: a reference to `module.name`, not resolved to anything.
    Global(String, String),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
    /// A boolean.
    Bool(bool),
    /// `None`.
    None,
    /// A tuple.
    Tuple(Vec<Id>),
    /// A list.
    List(Vec<Id>),
    /// A dict, as ordered key/value pairs.
    Dict(Vec<(Id, Id)>),
    /// The stack mark.
    Mark,
    /// `callable(*args)`, recorded and never run.
    Reduce(Id, Id),
    /// `BUILD`: state applied to an object, recorded and never applied.
    Build(Id, Id),
    /// A persistent id.
    PersId(Id),
}

/// The parsed pickle.
#[derive(Debug)]
pub struct Parsed {
    /// Every object created.
    pub arena: Vec<Node>,
    /// The object left by `STOP`.
    pub root: Id,
}

struct Vm<'a> {
    bytes: &'a [u8],
    pos: usize,
    op_at: usize,
    arena: Vec<Node>,
    stack: Vec<Id>,
    memo: HashMap<u32, Id>,
    ops: usize,
}

fn allowed(module: &str, name: &str) -> bool {
    ALLOWED_REDUCE_TARGETS
        .iter()
        .any(|&(m, n)| m == module && n == name)
}

impl<'a> Vm<'a> {
    fn malformed(&self, detail: &str) -> PickleError {
        PickleError::Malformed {
            offset: self.op_at,
            detail: detail.to_string(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], PickleError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or(PickleError::Truncated { offset: self.op_at })?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, PickleError> {
        Ok(self.take(1)?[0])
    }

    fn le(&mut self, n: usize) -> Result<u64, PickleError> {
        Ok(self
            .take(n)?
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b)))
    }

    fn alloc(&mut self, node: Node) -> Result<Id, PickleError> {
        if self.arena.len() >= MAX_NODES {
            return Err(PickleError::LimitExceeded {
                limit: "object count",
            });
        }
        self.arena.push(node);
        Ok(self.arena.len() - 1)
    }

    fn push(&mut self, node: Node) -> Result<(), PickleError> {
        let id = self.alloc(node)?;
        self.stack.push(id);
        Ok(())
    }

    fn pop(&mut self) -> Result<Id, PickleError> {
        self.stack
            .pop()
            .ok_or_else(|| self.malformed("empty stack"))
    }

    fn pop_n(&mut self, n: usize) -> Result<Vec<Id>, PickleError> {
        let at = self
            .stack
            .len()
            .checked_sub(n)
            .ok_or_else(|| self.malformed("stack underflow"))?;
        Ok(self.stack.split_off(at))
    }

    fn pop_to_mark(&mut self) -> Result<Vec<Id>, PickleError> {
        let mark = self
            .stack
            .iter()
            .rposition(|&i| self.arena[i] == Node::Mark)
            .ok_or_else(|| self.malformed("no mark on the stack"))?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(items)
    }

    fn line(&mut self) -> Result<String, PickleError> {
        let rest = &self.bytes[self.pos..];
        let n = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(PickleError::Truncated { offset: self.op_at })?;
        let s = String::from_utf8(rest[..n].to_vec())
            .map_err(|_| self.malformed("global name is not UTF-8"))?;
        self.pos += n + 1;
        Ok(s.trim_end_matches('\r').to_string())
    }

    fn string(&mut self, len: u64) -> Result<String, PickleError> {
        let n = usize::try_from(len).map_err(|_| self.malformed("string length"))?;
        let raw = self.take(n)?;
        String::from_utf8(raw.to_vec()).map_err(|_| self.malformed("string is not UTF-8"))
    }

    fn node(&self, id: Id) -> &Node {
        &self.arena[id]
    }

    fn global(&mut self, op: u8) -> Result<(), PickleError> {
        let (module, name) = if op == b'c' {
            (self.line()?, self.line()?)
        } else {
            let name = self.pop()?;
            let module = self.pop()?;
            match (self.node(module).clone(), self.node(name).clone()) {
                (Node::Str(m), Node::Str(n)) => (m, n),
                _ => return Err(self.malformed("STACK_GLOBAL needs two strings")),
            }
        };
        self.push(Node::Global(module, name))
    }

    fn constant(&mut self, op: u8) -> Result<(), PickleError> {
        match op {
            b'N' => self.push(Node::None),
            0x88 => self.push(Node::Bool(true)),
            _ => self.push(Node::Bool(false)),
        }
    }

    fn number(&mut self, op: u8) -> Result<(), PickleError> {
        let node = match op {
            b'K' => Node::Int(self.le(1)? as i64),
            b'M' => Node::Int(self.le(2)? as i64),
            b'J' => Node::Int(i64::from(self.le(4)? as u32 as i32)),
            b'G' => {
                let raw = self.take(8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(raw);
                Node::Float(f64::from_be_bytes(a))
            }
            _ => self.long1()?,
        };
        self.push(node)
    }

    /// `LONG1`: a length byte, then that many little-endian two's-complement
    /// bytes. Anything wider than 8 bytes is refused rather than truncated.
    fn long1(&mut self) -> Result<Node, PickleError> {
        let n = usize::from(self.u8()?);
        if n > 8 {
            return Err(self.malformed("integer wider than 64 bits"));
        }
        let raw = self.take(n)?;
        let mut v = raw.iter().rev().fold(0u64, |a, &b| (a << 8) | u64::from(b));
        if n > 0 && n < 8 && raw[n - 1] & 0x80 != 0 {
            v |= u64::MAX << (8 * n);
        }
        Ok(Node::Int(v as i64))
    }

    fn text(&mut self, op: u8) -> Result<(), PickleError> {
        let len = if op == b'X' { self.le(4)? } else { self.le(1)? };
        let s = self.string(len)?;
        self.push(Node::Str(s))
    }

    fn memo_op(&mut self, op: u8) -> Result<(), PickleError> {
        match op {
            b'q' | b'r' | 0x94 => {
                let key = match op {
                    b'q' => self.le(1)? as u32,
                    b'r' => self.le(4)? as u32,
                    _ => u32::try_from(self.memo.len())
                        .map_err(|_| PickleError::LimitExceeded { limit: "memo size" })?,
                };
                let top = *self
                    .stack
                    .last()
                    .ok_or_else(|| self.malformed("MEMOIZE on an empty stack"))?;
                self.memo.insert(key, top);
                Ok(())
            }
            _ => {
                let key = (if op == b'h' { self.le(1)? } else { self.le(4)? }) as u32;
                let id = *self
                    .memo
                    .get(&key)
                    .ok_or_else(|| self.malformed("GET of a missing memo entry"))?;
                self.stack.push(id);
                Ok(())
            }
        }
    }

    fn container(&mut self, op: u8) -> Result<(), PickleError> {
        let node = match op {
            b'(' => Node::Mark,
            b')' => Node::Tuple(Vec::new()),
            b']' => Node::List(Vec::new()),
            b'}' => Node::Dict(Vec::new()),
            b't' => Node::Tuple(self.pop_to_mark()?),
            b'l' => Node::List(self.pop_to_mark()?),
            b'd' => {
                let items = self.pop_to_mark()?;
                Node::Dict(self.pairs(&items)?)
            }
            _ => Node::Tuple(self.pop_n(usize::from(op - 0x84))?),
        };
        self.push(node)
    }

    fn pairs(&self, items: &[Id]) -> Result<Vec<(Id, Id)>, PickleError> {
        let (pairs, rest) = items.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(self.malformed("odd number of dict items"));
        }
        Ok(pairs.iter().map(|&[k, v]| (k, v)).collect())
    }

    /// `APPEND`/`APPENDS`/`SETITEM`/`SETITEMS`: extend the container below.
    fn fill(&mut self, op: u8) -> Result<(), PickleError> {
        let items = match op {
            b'a' => self.pop_n(1)?,
            b's' => self.pop_n(2)?,
            _ => self.pop_to_mark()?,
        };
        let target = *self
            .stack
            .last()
            .ok_or_else(|| self.malformed("no container to fill"))?;
        let pairs = if matches!(op, b's' | b'u') {
            Some(self.pairs(&items)?)
        } else {
            None
        };
        match (&mut self.arena[target], pairs) {
            (Node::List(l), None) => l.extend(items),
            (Node::Dict(d), Some(p)) => d.extend(p),
            _ => return Err(self.malformed("append or set on the wrong kind of object")),
        }
        Ok(())
    }

    fn check_callable(&self, id: Id) -> Result<(), PickleError> {
        match self.node(id) {
            Node::Global(m, n) if allowed(m, n) => Ok(()),
            Node::Global(m, n) => Err(PickleError::DisallowedCallable {
                module: m.clone(),
                name: n.clone(),
            }),
            _ => Err(self.malformed("callable is not a global")),
        }
    }

    fn object(&mut self, op: u8) -> Result<(), PickleError> {
        match op {
            b'R' | 0x81 => {
                let args = self.pop()?;
                let callable = self.pop()?;
                self.check_callable(callable)?;
                if self.node(callable) == &Node::Global("collections".into(), "OrderedDict".into())
                {
                    // `OrderedDict()` is the empty dict that SETITEMS then fills.
                    return self.push(Node::Dict(Vec::new()));
                }
                self.push(Node::Reduce(callable, args))
            }
            b'b' => {
                let state = self.pop()?;
                let obj = self.pop()?;
                self.push(Node::Build(obj, state))
            }
            _ => {
                let id = self.pop()?;
                self.push(Node::PersId(id))
            }
        }
    }

    /// Run one opcode. Returns the root when `STOP` is reached.
    fn step(&mut self) -> Result<Option<Id>, PickleError> {
        self.op_at = self.pos;
        let op = self.u8()?;
        match op {
            0x80 => drop(self.u8()?),
            0x95 => drop(self.take(8)?),
            b'.' => return Ok(Some(self.pop()?)),
            b'c' | 0x93 => self.global(op)?,
            b'N' | 0x88 | 0x89 => self.constant(op)?,
            b'K' | b'M' | b'J' | b'G' | 0x8a => self.number(op)?,
            b'X' | 0x8c => self.text(op)?,
            b'q' | b'r' | b'h' | b'j' | 0x94 => self.memo_op(op)?,
            b'(' | b')' | b']' | b'}' | b't' | b'l' | b'd' | 0x85..=0x87 => self.container(op)?,
            b'a' | b's' | b'e' | b'u' => self.fill(op)?,
            b'R' | 0x81 | b'b' | b'Q' => self.object(op)?,
            _ => {
                return Err(PickleError::UnsupportedOpcode {
                    opcode: op,
                    offset: self.op_at,
                });
            }
        }
        Ok(None)
    }
}

/// Parse a pickle into an object arena without executing it.
///
/// # Errors
///
/// [`PickleError`]: truncation, an unsupported opcode, a structural fault, a
/// callable outside [`ALLOWED_REDUCE_TARGETS`], or a size limit.
pub fn parse_pickle(bytes: &[u8]) -> Result<Parsed, PickleError> {
    let mut vm = Vm {
        bytes,
        pos: 0,
        op_at: 0,
        arena: Vec::new(),
        stack: Vec::new(),
        memo: HashMap::new(),
        ops: 0,
    };
    loop {
        vm.ops += 1;
        if vm.ops > MAX_OPS {
            return Err(PickleError::LimitExceeded {
                limit: "opcode count",
            });
        }
        if let Some(root) = vm.step()? {
            return Ok(Parsed {
                arena: vm.arena,
                root,
            });
        }
    }
}
