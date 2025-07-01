// RUN: iree-opt %s

// PDL pattern spec to annotate operations with specialization ranges.

pdl.pattern @f16_pingpong : benefit(1) {
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]
  %elemtypes = pdl.attribute = [f16, f16, f32]
  %operands = pdl.operands
  %types = pdl.types
  %matmul = pdl.operation (%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
  pdl.apply_native_constraint "matchContraction"(
        %matmul, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  // Skip if the operation already has ranges.
  %attr_name = pdl.attribute = "iree_codegen.specialization_ranges"
  pdl.apply_native_constraint "hasAttr"(
        %matmul, %attr_name
        : !pdl.operation, !pdl.attribute) {isNegated = true}

  pdl.rewrite %matmul {
    %ranges = pdl.attribute = #util<int.assumption.multi_array[
        [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 64>], // Large pingpong
        [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 64>]  // Medium pingpong
      ]>
    pdl.apply_native_rewrite "annotateOperation"(
        %matmul, %attr_name, %ranges
        : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

pdl.pattern @f8E4M3_pingpong : benefit(1) {
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]
  %elemtypes = pdl.attribute = [f8E4M3FNUZ, f8E4M3FNUZ, f32]
  %operands = pdl.operands
  %types = pdl.types
  %matmul = pdl.operation (%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
  pdl.apply_native_constraint "matchContraction"(
        %matmul, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  // Skip if the operation already has ranges.
  %attr_name = pdl.attribute = "iree_codegen.specialization_ranges"
  pdl.apply_native_constraint "hasAttr"(
        %matmul, %attr_name
        : !pdl.operation, !pdl.attribute) {isNegated = true}

  pdl.rewrite %matmul {
    %ranges = pdl.attribute = #util<int.assumption.multi_array[
        [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 128>], // Large pingpong
        [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 128>]  // Medium pingpong
      ]>
    pdl.apply_native_rewrite "annotateOperation"(
        %matmul, %attr_name, %ranges
        : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

pdl.pattern @attention : benefit(1) {
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d6, d1, d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d6, d1, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d6)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
  ]
  %operands = pdl.operands
  %types = pdl.types
  %attention = pdl.operation "iree_linalg_ext.attention"
    (%operands : !pdl.range<value>) {"indexing_maps" = %imaps} -> (%types : !pdl.range<type>)

  // Skip if the operation already has ranges.
  %attr_name = pdl.attribute = "iree_codegen.specialization_ranges"
  pdl.apply_native_constraint "hasAttr"(
        %attention, %attr_name
        : !pdl.operation, !pdl.attribute) {isNegated = true}

  pdl.rewrite %attention {
    %ranges = pdl.attribute = #util<int.assumption.multi_array[
        [<udiv = 1>,
         <udiv = 1>,
         <udiv = 1>,
         <udiv = 64>,
         <udiv = 1>,
         <udiv = 1>,
         <udiv = 64>]
      ]>
    pdl.apply_native_rewrite "annotateOperation"(
        %attention, %attr_name, %ranges
        : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}
