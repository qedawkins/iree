// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @generic(%0: tensor<?xi32>, %1: tensor<?x?xi32>, %d0: index, %d1: index) {
  %2:4 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, tensor<?xi32>{%d0}, tensor<?x?xi32>{%d0, %d1}, tensor<?x?xi32>) {
    util.optimization_barrier %id, %n, %ref, %ref_1, %ref_2, %ref_3 : index, index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>
    pcf.return
  } {hello = "world"}
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, tensor<?x?xi32>
  util.return
}

// CHECK-LABEL: @generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ARG0]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ARG1]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[N:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<?xi32>, tensor<?xi32>{%[[D0]]}, tensor<?x?xi32>{%[[D0]], %[[D1]]}, tensor<?x?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[ID]], %[[N]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }  {hello = "world"}

// -----

util.func private @generic_no_inits() {
  pcf.generic scope(#pcf.test_scope)
    execute[%id: index, %n: index] {
    util.optimization_barrier %id, %n : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_no_inits

//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute[%[[ID:.+]]: index, %[[N:.+]]: index] {
//  CHECK-NEXT:       util.optimization_barrier %[[ID]], %[[N]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:   }

// -----

util.func private @generic_readonly(%src: tensor<4xf32>, %dst: tensor<4xf32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%src_ref <- %src : tensor<4xf32>, %dst_ref = %dst)[%id: index, %n: index]
         : (!pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>)
        -> (tensor<4xf32>) {
    util.optimization_barrier %id, %n, %src_ref, %dst_ref : index, index, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4xf32>
  util.return
}

// CHECK-LABEL: @generic_readonly
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<4xf32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: tensor<4xf32>
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[SRC_REF:[A-Za-z0-9_]+]] <- %[[SRC]] : tensor<4xf32>,
//  CHECK-SAME:             %[[DST_REF:[A-Za-z0-9_]+]] = %[[DST]])
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<4xf32>) {

// -----

util.func private @generic_interleaved_readonly(%src: tensor<4xf32>, %dst: tensor<?xf32>, %d0: index) {
  %0:2 = pcf.generic scope(#pcf.test_scope)
    execute(%dst_ref = %dst, %src_ref <- %src : tensor<4xf32>, %tmp_ref)[%id: index, %n: index]
         : (!pcf.sref<?xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<?xf32, #pcf.test_scope>)
        -> (tensor<?xf32>, tensor<?xf32>{%d0}) {
    util.optimization_barrier %id, %n, %dst_ref, %src_ref, %tmp_ref : index, index, !pcf.sref<?xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<?xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
  util.return
}

// CHECK-LABEL: @generic_interleaved_readonly
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<4xf32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[DST_REF:[A-Za-z0-9_]+]] = %[[DST]],
//  CHECK-SAME:             %[[SRC_REF:[A-Za-z0-9_]+]] <- %[[SRC]] : tensor<4xf32>,
//  CHECK-SAME:             %[[TMP_REF:[A-Za-z0-9_]+]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[N:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<4xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<?xf32>, tensor<?xf32>{%[[D0]]}) {
//  CHECK-NEXT:       util.optimization_barrier %[[ID]], %[[N]], %[[DST_REF]], %[[SRC_REF]], %[[TMP_REF]]

// -----

util.func private @generic_memref(%0: memref<?xi32>, %1: memref<?x?xi32>, %d0: index, %d1: index) {
  %2:4 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id:index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>)
        -> (memref<?xi32>, memref<?xi32>{%d0}, memref<?x?xi32>{%d0, %d1}, memref<?x?xi32>) {
    util.optimization_barrier %id, %n, %ref, %ref_1, %ref_2, %ref_3 : index, index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : memref<?xi32>, memref<?xi32>, memref<?x?xi32>, memref<?x?xi32>
  util.return
}

// CHECK-LABEL: @generic_memref
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: memref<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ARG0]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ARG1]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[N:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D0]]}, memref<?x?xi32>{%[[D0]], %[[D1]]}, memref<?x?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[ID]], %[[N]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @generic_with_initializer(%arg0: memref<?x?xi32>) {
  %0 = pcf.generic scope(#pcf.test_scope) initialize {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } -> (%i: index)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.test_scope>)
        -> (memref<?x?xi32>) {
    util.optimization_barrier %i : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_with_initializer
//       CHECK:   pcf.generic scope(#pcf.test_scope) initialize {
//  CHECK-NEXT:       %[[C42:.+]] = arith.constant 42
//  CHECK-NEXT:       pcf.yield %[[C42]]
//  CHECK-NEXT:     } -> (%[[I:.+]]: index)
//  CHECK-NEXT:     execute(%{{.*}} = %{{.*}})
//  CHECK-SAME:             [{{.*}}]
//  CHECK-NEXT:          : (!pcf.sref<?x?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?x?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[I]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @loop(%0: tensor<?xi32>, %1: tensor<?x?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, tensor<?xi32>{%d0}, tensor<?x?xi32>{%d0, %d1}, tensor<?x?xi32>) {
    util.optimization_barrier %id, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>
    pcf.return
  } {hello = "world"}
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, tensor<?x?xi32>
  util.return
}

// CHECK-LABEL: @loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ARG0]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ARG1]])
//  CHECK-SAME:             [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<?xi32>, tensor<?xi32>{%[[D0]]}, tensor<?x?xi32>{%[[D0]], %[[D1]]}, tensor<?x?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }  {hello = "world"}

// -----

util.func private @loop_no_inits(%n: index) {
  pcf.loop scope(#pcf.test_scope) count(%n)
    execute[%num_threads: index] {
    util.optimization_barrier %num_threads : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @loop_no_inits

//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%{{.*}})
//  CHECK-NEXT:     execute[%[[NUM_THREADS:.+]]: index] {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:   }

// -----

util.func private @loop_readonly(%src: tensor<4xf32>, %dst: tensor<4xf32>, %n: index) {
  %0 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%src_ref <- %src : tensor<4xf32>, %dst_ref = %dst)[%id: index]
         : (!pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>)
        -> (tensor<4xf32>) {
    util.optimization_barrier %id, %src_ref, %dst_ref : index, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4xf32>
  util.return
}

// CHECK-LABEL: @loop_readonly
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<4xf32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: tensor<4xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index
//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[SRC_REF:[A-Za-z0-9_]+]] <- %[[SRC]] : tensor<4xf32>,
//  CHECK-SAME:             %[[DST_REF:[A-Za-z0-9_]+]] = %[[DST]])
//  CHECK-SAME:             [%{{.*}}: index]

// -----

util.func private @loop_interleaved_readonly(%src: tensor<4xf32>, %dst: tensor<?xf32>, %d0: index, %n: index) {
  %0:2 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%dst_ref = %dst, %src_ref <- %src : tensor<4xf32>, %tmp_ref)[%id: index]
         : (!pcf.sref<?xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<?xf32, #pcf.test_scope>)
        -> (tensor<?xf32>, tensor<?xf32>{%d0}) {
    util.optimization_barrier %id, %dst_ref, %src_ref, %tmp_ref : index, !pcf.sref<?xf32, #pcf.test_scope>, !pcf.sref<4xf32, #pcf.test_scope>, !pcf.sref<?xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
  util.return
}

// CHECK-LABEL: @loop_interleaved_readonly
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<4xf32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index
//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[DST_REF:[A-Za-z0-9_]+]] = %[[DST]],
//  CHECK-SAME:             %[[SRC_REF:[A-Za-z0-9_]+]] <- %[[SRC]] : tensor<4xf32>,
//  CHECK-SAME:             %[[TMP_REF:[A-Za-z0-9_]+]])
//  CHECK-SAME:             [%[[ID:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<4xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<?xf32>, tensor<?xf32>{%[[D0]]}) {
//  CHECK-NEXT:       util.optimization_barrier %[[ID]], %[[DST_REF]], %[[SRC_REF]], %[[TMP_REF]]

// -----

util.func private @loop_memref(%0: memref<?xi32>, %1: memref<?x?xi32>, %d0: index, %d1: index, %n: index) {
  %2:4 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>)
        -> (memref<?xi32>, memref<?xi32>{%d0}, memref<?x?xi32>{%d0, %d1}, memref<?x?xi32>) {
    util.optimization_barrier %id, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>, !pcf.sref<?x?xi32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : memref<?xi32>, memref<?xi32>, memref<?x?xi32>, memref<?x?xi32>
  util.return
}

// CHECK-LABEL: @loop_memref
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: memref<?xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index

//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ARG0]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ARG1]])
//  CHECK-SAME:             [%[[NUM_THREADS:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D0]]}, memref<?x?xi32>{%[[D0]], %[[D1]]}, memref<?x?xi32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[NUM_THREADS]], %[[REF]], %[[REF1]], %[[REF2]], %[[REF3]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }
