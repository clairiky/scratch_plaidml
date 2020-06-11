// pmlc-opt lcr_tests/test.mlir -convert-std-to-llvm -target-cpu | pmlc-jit

#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>


module {
  func @dot(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> tensor<f32>
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %cf1 = constant 1.00000e+00 : f32
    return %0 : tensor<2x2xf32>
   // return %cf1 : f32
  }
  func @simpleLoad(%A: memref<20x10xf32>, %i: index, %j: index) -> (f32) {
    %0 = load %A[%i, %j] : memref<20x10xf32>
    return %0: f32
  }
  
//   func @main() {
//     %i = constant 0 : index
//     %j = constant 10 : index // out of bounds
//     %buf = alloc() : memref<20x10xf32>
//     call @simpleLoad(%buf, %i, %j) : (memref<20x10xf32>, index, index) -> (f32)
//     dealloc %buf : memref<20x10xf32>
//     return
//   } 
  func @main() {
  %A = alloc() : memref<2x2xf32>
  %B = alloc() : memref<2x2xf32>
  %C = alloc() : memref<2x2xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<2x2xf32>, f32
  linalg.fill(%B, %cf1) : memref<2x2xf32>, f32
  linalg.fill(%C, %cf1) : memref<2x2xf32>, f32

//   /%0 = tile.constant dense<[[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tensor<2x2xf64>
//   %1 = tile.reshape(%0 : tensor<2x3xf64>) to tensor<2x2xf64>
//   %2 = tile.constant dense<[1.000000e+00, 2.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<4xf64>
//   %3 = tile.reshape(%2 : tensor<4xf64>) to tensor<2x2xf64>
  //
//call @dot(%E, %D) : (memref<2x2xf32>, memref<2x2xf32>) -> (memref<2x2xf32>)
//  call @dot(%E, %D) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)

//   %i = constant 0 : index
//   %j = constant 10 : index // out of bounds
//   %buf = alloc() : memref<20x10xf32>
//   call @simpleLoad(%buf, %i, %j) : (memref<20x10xf32>, index, index) -> (f32)
//   dealloc %buf : memref<20x10xf32>
  return
  }
}
