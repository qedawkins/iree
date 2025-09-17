// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @generic(%0: tensor<?xi32>, %1: tensor<?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.generic scope(#pcf.sequential) tripcount(%n)
    initialize(%ref = %0, %ref_1[%token: !pcf.token<#pcf.sequential>], %ref_2, %ref_3[%token_1: !pcf.token<#pcf.sequential>] = %1)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>)
           -> (tensor<?xi32>, tensor<?xi32>{%d0}, tensor<?xi32>{%d1}, tensor<?xi32>) {
    util.optimization_barrier %num_threads, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>, !pcf.sref<?xi32, #pcf.sequential>
    pcf.join_yield %token, %token_1 : !pcf.token<#pcf.sequential>, !pcf.token<#pcf.sequential>
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

//       CHECK:   pcf.generic scope(#pcf.sequential) tripcount(%[[N]])
//  CHECK-NEXT:     initialize(%[[REF:.+]] = %[[ARG0]],
//  CHECK-SAME:                %[[REF1:.+]][%[[TOKEN:.+]]: !pcf.token<#pcf.sequential>],
//  CHECK-SAME:                %[[REF2:.+]],
//  CHECK-SAME:                %[[REF3:.+]][%[[TOKEN1:.+]]: !pcf.token<#pcf.sequential>] = %[[ARG1]])
//  CHECK-SAME:                [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:             : (!pcf.sref<?xi32, #pcf.sequential>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.sequential>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.sequential>,
//  CHECK-SAME:                !pcf.sref<?xi32, #pcf.sequential>)
//  CHECK-NEXT:            -> (tensor<?xi32>, tensor<?xi32>{%[[D0]]}, tensor<?xi32>{%[[D1]]}, tensor<?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.join_yield %[[TOKEN]], %[[TOKEN1]]
//  CHECK-NEXT:     }  {hello = "world"}

// -----

util.func private @generic_no_inits() {
  pcf.generic scope(#pcf.sequential)
    initialize[%num_threads: index] {
    util.optimization_barrier %num_threads : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_no_inits

//       CHECK:   pcf.generic scope(#pcf.sequential)
//  CHECK-NEXT:     initialize[%[[NUM_THREADS:.+]]: index] {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:   }
