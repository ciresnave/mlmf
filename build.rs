use std::env;
use std::path::Path;

fn main() {
    // Only compile protobuf when ONNX feature is enabled
    if env::var("CARGO_FEATURE_ONNX").is_ok() {
        let out_dir = env::var("OUT_DIR").unwrap();

        // ONNX protobuf schema (simplified version)
        let onnx_proto = r#"
syntax = "proto3";

package onnx;

message ModelProto {
  optional int64 ir_version = 1;
  repeated string opset_import = 2;
  optional string producer_name = 3;
  optional string producer_version = 4;
  optional string domain = 5;
  optional int64 model_version = 6;
  optional string doc_string = 7;
  optional GraphProto graph = 8;
  repeated string metadata_props = 14;
}

message GraphProto {
  repeated NodeProto node = 1;
  string name = 2;
  repeated ValueInfoProto initializer = 5;
  optional string doc_string = 10;
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;
  repeated TensorProto initializer_tensor = 15;
}

message NodeProto {
  repeated string input = 1;
  repeated string output = 2;
  optional string name = 3;
  optional string op_type = 4;
  optional string domain = 5;
  repeated AttributeProto attribute = 6;
  optional string doc_string = 7;
}

message ValueInfoProto {
  optional string name = 1;
  optional TypeProto type = 2;
  optional string doc_string = 3;
}

message TypeProto {
  oneof value {
    TensorTypeProto tensor_type = 1;
  }
}

message TensorTypeProto {
  optional int32 elem_type = 1;
  optional TensorShapeProto shape = 2;
}

message TensorShapeProto {
  repeated Dimension dim = 1;

  message Dimension {
    oneof value {
      int64 dim_value = 1;
      string dim_param = 2;
    }
    optional string denotation = 3;
  }
}

message TensorProto {
  repeated int64 dims = 1;
  optional int32 data_type = 2;
  optional SegmentProto segment = 3;
  repeated float float_data = 4;
  repeated int32 int32_data = 5;
  repeated bytes string_data = 6;
  repeated int64 int64_data = 7;
  optional string name = 8;
  optional string doc_string = 12;
  optional bytes raw_data = 9;
  repeated StringStringEntryProto external_data = 13;
  optional DataLocation data_location = 14;
  repeated double double_data = 10;
  repeated uint64 uint64_data = 11;
}

message SegmentProto {
  optional int64 begin = 1;
  optional int64 end = 2;
}

message StringStringEntryProto {
  optional string key = 1;
  optional string value = 2;
}

enum DataLocation {
  DEFAULT = 0;
  EXTERNAL = 1;
}

message AttributeProto {
  optional string name = 1;
  optional string ref_attr_name = 21;
  optional string doc_string = 13;
  optional AttributeType type = 20;
  optional float f = 2;
  optional int64 i = 3;
  optional bytes s = 4;
  optional TensorProto t = 5;
  optional GraphProto g = 6;
  optional SparseTensorProto sparse_tensor = 22;
  optional TypeProto tp = 14;
  repeated float floats = 7;
  repeated int64 ints = 8;
  repeated bytes strings = 9;
  repeated TensorProto tensors = 10;
  repeated GraphProto graphs = 11;
  repeated SparseTensorProto sparse_tensors = 23;
  repeated TypeProto type_protos = 15;
}

enum AttributeType {
  UNDEFINED = 0;
  FLOAT = 1;
  INT = 2;
  STRING = 3;
  TENSOR = 4;
  GRAPH = 5;
  SPARSE_TENSOR = 11;
  TYPE_PROTO = 13;
  FLOATS = 6;
  INTS = 7;
  STRINGS = 8;
  TENSORS = 9;
  GRAPHS = 10;
  SPARSE_TENSORS = 12;
  TYPE_PROTOS = 14;
}

message SparseTensorProto {
  optional TensorProto values = 1;
  optional TensorProto indices = 2;
  repeated int64 dims = 3;
}
"#;

        // Write the proto file
        let proto_path = Path::new(&out_dir).join("onnx.proto");
        std::fs::write(&proto_path, onnx_proto).unwrap();

        // Generate Rust code using prost
        prost_build::Config::new()
            .out_dir(&out_dir)
            .compile_protos(&[&proto_path], &[Path::new(&out_dir)])
            .unwrap();

        println!("cargo:rerun-if-changed=build.rs");
    }
}
