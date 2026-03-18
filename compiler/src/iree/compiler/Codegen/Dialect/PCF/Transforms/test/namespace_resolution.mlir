// RUN: iree-opt --iree-pcf-test-namespace-resolution --allow-unregistered-dialect --verify-diagnostics --split-input-file %s

// Leaf-only resolution (innermost anonymous namespace wins).
util.func private @leaf_resolution(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // expected-remark @below {{resolved '#pcf.ns_symleft' to !pcf.cluster<#pcf.test_scope, (0 -> d0), left>}}
    pcf.shared_executor.run_thread(%left)[%k]
        ()[%tid: index] {
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Qualified resolution through one named namespace.
util.func private @qualified_resolution(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), tg.right>) {
    // expected-remark @below {{resolved '#pcf.ns_symtg.left' to !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>}}
    pcf.shared_executor.run_thread(%left)[%k]
        ()[%tid: index] {
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Qualified resolution through nested namespaces.
util.func private @nested_resolution(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    // The inner tile_group takes %left (a cluster) as source, so
    // resolution also fires on the tile_group op itself.
    // expected-remark @below {{resolved '#pcf.ns_symouter.left' to !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>}}
    pcf.shared_executor.tile_group %left ns(inner) split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), inner.bot>) {
      // Resolves "outer.right" from inside the inner namespace.
      // expected-remark @below {{resolved '#pcf.ns_symouter.right' to !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>}}
      pcf.shared_executor.run_thread(%right)[%k]
          ()[%tid: index] {
        pcf.cluster_yield
      } : (!pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>)
      // Resolves "inner.top" from innermost namespace.
      // expected-remark @below {{resolved '#pcf.ns_syminner.top' to !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>}}
      pcf.shared_executor.run_thread(%top)[%j]
          ()[%tid: index] {
        pcf.cluster_yield
      } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>)
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Deeply nested qualified resolution (3-segment path: outer.inner.top).
util.func private @deep_qualified_resolution(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    // expected-remark @below {{resolved '#pcf.ns_symouter.left' to !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>}}
    pcf.shared_executor.tile_group %left ns(inner) split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), inner.bot>) {
      // Resolve "outer.inner.top" -- a 3-segment qualified path.
      // expected-remark @below {{resolved '#pcf.ns_symouter.inner.top' to !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>}}
      "test.dummy"() {test.resolve = "outer.inner.top"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Resolution via test.resolve string attribute.
util.func private @test_attr_resolution(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), tg.right>) {
    // expected-remark @below {{resolved '#pcf.ns_symtg.right' to !pcf.cluster<#pcf.test_scope, (d0 -> s0), tg.right>}}
    "test.dummy"() {test.resolve = "tg.right"} : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}
