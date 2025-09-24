// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @generic(%0: tensor<?xi32>, %1: tensor<?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.generic scope(#pcf.dummy_scope) tripcount(%n)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.dummy_scope>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.dummy_scope>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>, tensor<?xi32>{%d0}, tensor<?xi32>{%d1}, tensor<?xi32>) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.dummy_scope>, !pcf.token<#pcf.dummy_scope>
  } {hello = "world"}
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
  util.return
}

// CHECK-LABEL: @generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.generic scope(#pcf.dummy_scope) tripcount(%[[N]])
//  CHECK-NEXT:     initialize(%[[REF:.+]] = %[[ARG0]],
//  CHECK-SAME:                %[[REF1:.+]][%[[TOKEN:.+]]: !pcf.token<#pcf.dummy_scope>],
//  CHECK-SAME:                %[[REF2:.+]],
//  CHECK-SAME:                %[[REF3:.+]][%[[TOKEN1:.+]]: !pcf.token<#pcf.dummy_scope>] = %[[ARG1]])
//  CHECK-SAME:                [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:             : (!pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>)
//  CHECK-NEXT:            -> (tensor<?xi32>, tensor<?xi32>{%[[D0]]}, tensor<?xi32>{%[[D1]]}, tensor<?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.join_yield %[[TOKEN]], %[[TOKEN1]]
//  CHECK-NEXT:     }  {hello = "world"}

// -----

util.func private @generic_no_inits() {
  pcf.generic scope(#pcf.dummy_scope)
    initialize[%num_threads: index] {
    util.optimization_barrier %num_threads : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_no_inits

//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     initialize[%[[NUM_THREADS:.+]]: index] {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:   }

// -----

util.func private @generic_memref(%0: memref<?xi32>, %1: memref<?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.generic scope(#pcf.dummy_scope) tripcount(%n)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.dummy_scope>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.dummy_scope>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (memref<?xi32>, memref<?xi32>{%d0}, memref<?xi32>{%d1}, memref<?xi32>) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?xi32, #pcf.dummy_scope>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.dummy_scope>, !pcf.token<#pcf.dummy_scope>
  }
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>
  util.return
}

// CHECK-LABEL: @generic_memref
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: memref<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: memref<?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.generic scope(#pcf.dummy_scope) tripcount(%[[N]])
//  CHECK-NEXT:     initialize(%[[REF:.+]] = %[[ARG0]],
//  CHECK-SAME:                %[[REF1:.+]][%[[TOKEN:.+]]: !pcf.token<#pcf.dummy_scope>],
//  CHECK-SAME:                %[[REF2:.+]],
//  CHECK-SAME:                %[[REF3:.+]][%[[TOKEN1:.+]]: !pcf.token<#pcf.dummy_scope>] = %[[ARG1]])
//  CHECK-SAME:                [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:             : (!pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.dummy_scope>)
//  CHECK-NEXT:            -> (memref<?xi32>, memref<?xi32>{%[[D0]]}, memref<?xi32>{%[[D1]]}, memref<?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.join_yield %[[TOKEN]], %[[TOKEN1]]
//  CHECK-NEXT:     }
