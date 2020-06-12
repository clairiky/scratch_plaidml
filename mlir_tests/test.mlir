#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>


module {
  func @mdot(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> tensor<f32>
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {idxs = ["i", "j", "k"], sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %cf1 = constant 1.00000e+00 : f32
    return %0 : tensor<2x2xf32>
   //return %cf1 : f32
  }

 func @test(%arg0: tensor<2x2xf32>) -> f32 {
    %cst = constant 1.000000e+00 : f32
    return %cst : f32
  }
  func @main() {
    %cst = constant 1.000000e+00 : f32
    %0 = alloc() : memref<2x2xf32>
    %1 = alloc() : memref<2x2xf32>


  //linalg.fill(%A, %cf1) : memref<2x2xf32>, f32


   affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        affine.store %cst, %0[%arg0, %arg1] : memref<2x2xf32>
      }
    }
  //linalg.fill(%B, %cf1) : memref<2x2xf32>, f32
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        affine.store %cst, %1[%arg0, %arg1] : memref<2x2xf32>
      }
    }


  %E = std.tensor_load %0 : memref<2x2xf32>
  %D = std.tensor_load %1 : memref<2x2xf32>

  call @mdot(%E, %D) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  //call @test(%D) : (tensor<2x2xf32>) -> (f32)
  //%c0 = "std.constant"() {value = 0: index} : () -> index
 
  //%0 = tensor_from_elements(%c0) : tensor<1xindex>
  //%2 = tensor_cast %0 : tensor<1xf32> to tensor<2x2xf32>
  return
  }
}