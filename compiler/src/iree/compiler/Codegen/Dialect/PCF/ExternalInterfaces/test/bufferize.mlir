// RUN: iree-opt %s --one-shot-bufferize --split-input-file | FileCheck %s

util.func private @bufferize_generic(%d0: index, %d1: index, %d2: index, %d3: index) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %1 = bufferization.alloc_tensor(%d3) {memory_space = "foo"} : tensor<?xi32>
  %2:4 = pcf.generic scope(#pcf.dummy_scope)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.dummy_scope>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.dummy_scope>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>, tensor<?xi32>{%d1}, tensor<?xi32>{%d2}, tensor<?xi32>) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.dummy_scope>, !pcf.token<#pcf.dummy_scope>
  }
  util.return
}

// CHECK-LABEL: @bufferize_generic(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D3:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//   CHECK-DAG:   %[[ALLOC1:.+]] = memref.alloc(%[[D3]]) {alignment = 64 : i64} : memref<?xi32, "foo">
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     initialize(%[[REF:.+]] = %[[ALLOC]],
//  CHECK-SAME:                %[[REF1:.+]][%[[TOKEN:.+]]: !pcf.token<#pcf.dummy_scope>],
//  CHECK-SAME:                %[[REF2:.+]],
//  CHECK-SAME:                %[[REF3:.+]][%[[TOKEN1:.+]]: !pcf.token<#pcf.dummy_scope>] = %[[ALLOC1]])
//  CHECK-SAME:                [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:             : (!pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>)
//  CHECK-NEXT:            -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.join_yield %[[TOKEN]], %[[TOKEN1]]
//  CHECK-NEXT:     }

// -----

util.func private @replay_bufferize_generic(%0: memref<?xi32>, %1: memref<?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.generic scope(#pcf.dummy_scope) count(%n)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.dummy_scope>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.dummy_scope>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (memref<?xi32>, memref<?xi32>{%d0}, memref<?xi32>{%d1}, memref<?xi32>) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.dummy_scope>, !pcf.token<#pcf.dummy_scope>
  }
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>
  util.return
}

// Verify that replaying bufferization works.
// CHECK-LABEL: @replay_bufferize_generic(
//       CHECK:   pcf.generic scope(#pcf.dummy_scope) count
//       CHECK:            -> (memref<?xi32>, memref<?xi32>{%{{.*}}}, memref<?xi32>{%{{.*}}}, memref<?xi32>) {

// -----

util.func private @bufferize_generic_mixed(%d0: index, %d1: index, %d2: index, %1: memref<?xi32, "foo">) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %2:4 = pcf.generic scope(#pcf.dummy_scope)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.dummy_scope>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.dummy_scope>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>, memref<?xi32>{%d1}, tensor<?xi32>{%d2}, memref<?xi32, "foo">) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.dummy_scope>, !pcf.token<#pcf.dummy_scope>
  }
  util.return
}

// CHECK-LABEL: @bufferize_generic_mixed(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[INIT1:[A-Za-z0-9]+]]: memref<?xi32, "foo">

//       CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     initialize(%[[REF:.+]] = %[[ALLOC]],
//  CHECK-SAME:                %[[REF1:.+]][%[[TOKEN:.+]]: !pcf.token<#pcf.dummy_scope>],
//  CHECK-SAME:                %[[REF2:.+]],
//  CHECK-SAME:                %[[REF3:.+]][%[[TOKEN1:.+]]: !pcf.token<#pcf.dummy_scope>] = %[[INIT1]])
//  CHECK-SAME:                [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:             : (!pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>)
//  CHECK-NEXT:            -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.join_yield %[[TOKEN]], %[[TOKEN1]]
//  CHECK-NEXT:     }

// -----

util.func private @write_tensor(%dst: !pcf.sref<?xi32, #pcf.dummy_scope>) {
  %src = bufferization.alloc_tensor() : tensor<2xi32>
  pcf.write_slice %src into %dst[1] [2] [1] : tensor<2xi32> into !pcf.sref<?xi32, #pcf.dummy_scope>
  util.return
}

// CHECK-LABEL: @write_tensor
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: !pcf.sref<?xi32, #pcf.dummy_scope>
//       CHECK:   %[[SRC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
//  CHECK-NEXT:   pcf.write_slice %[[SRC]] into %[[DST]][1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.dummy_scope>

// -----

util.func private @replay_write_tensor_bufferize(%src: memref<2xi32>, %dst: !pcf.sref<?xi32, #pcf.dummy_scope>) {
  pcf.write_slice %src into %dst[1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.dummy_scope>
  util.return
}

// CHECK-LABEL: @replay_write_tensor_bufferize
//  CHECK-NEXT:   pcf.write_slice %{{.*}} into %{{.*}}[1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.dummy_scope>
