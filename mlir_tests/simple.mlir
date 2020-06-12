// RUN: pmlc-opt -stdx-check-bounds -convert-std-to-llvm %s | pmlc-jit | FileCheck %s

module {
//   func @simpleLoad(%A: memref<2x2xf32>, %i: index, %j: index) -> (f32) {
//     %0 = load %A[%i, %j] : memref<2x2xf32>
//     return %0: f32
//   }

  func @addTensors(%A: tensor<2x2xf32>) -> (f32) {
    //%cst = constant 1.000000e+00 : f32
    //%x = addf %A, %A : tensor<2x2xf32>
    %cst = constant 1.000000e+00 : f32
    return %cst : f32
    //return %x : tensor<2x2xf32>
  }
  
  func @main() {
    %i = constant 0 : index
    %j = constant 9 : index 
    %buf = alloc() : memref<2x2xf32>
    %cst = constant 1.000000e+00 : f32

    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        affine.store %cst, %buf[%arg0, %arg1] : memref<2x2xf32>
      }
    }
    %D = std.tensor_load %buf : memref<2x2xf32>

    %E = call @addTensors(%D) : (tensor<2x2xf32>) -> (f32)
    //call @simpleLoad(%buf, %i, %j) : (memref<2x2xf32>, index, index) -> (f32)
    
    dealloc %buf : memref<2x2xf32>
    return
  }
  // CHECK: ERROR: of bounds index for mlir::LoadOp or mlir::StoreOp
}