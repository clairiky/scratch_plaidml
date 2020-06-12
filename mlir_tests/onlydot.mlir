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
}