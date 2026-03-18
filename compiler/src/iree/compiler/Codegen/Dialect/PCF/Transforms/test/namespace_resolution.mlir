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

// -----

// Leaf-only resolution with multiple nested anonymous namespaces.
// The inner namespace defines "left" which shadows the outer "left".
util.func private @leaf_inner_shadows_outer(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // expected-remark @below {{resolved '#pcf.ns_symleft' to !pcf.cluster<#pcf.test_scope, (0 -> d0), left>}}
    pcf.shared_executor.tile_group %left split [[%j]]
        (%left2: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), bot>) {
      // Resolves "left" to the inner namespace's "left", not the outer one.
      // expected-remark @below {{resolved '#pcf.ns_symleft' to !pcf.cluster<#pcf.test_scope, (0 -> d0), left>}}
      "test.dummy"() {test.resolve = "left"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Leaf-only resolution where inner namespace lacks the symbol.
// Falls through to the outer anonymous namespace.
util.func private @leaf_fallthrough_to_outer(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // expected-remark @below {{resolved '#pcf.ns_symleft' to !pcf.cluster<#pcf.test_scope, (0 -> d0), left>}}
    pcf.shared_executor.tile_group %left split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), bot>) {
      // "right" not in inner, falls through to outer.
      // expected-remark @below {{resolved '#pcf.ns_symright' to !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>}}
      "test.dummy"() {test.resolve = "right"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Mixed named and anonymous namespaces in a nesting chain.
// Outer is named, inner is anonymous. Leaf-only resolution from the
// anonymous inner still works for its own symbols.
util.func private @mixed_named_and_anonymous(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    // expected-remark @below {{resolved '#pcf.ns_symouter.left' to !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>}}
    pcf.shared_executor.tile_group %left split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), bot>) {
      // Resolve leaf-only from anonymous inner namespace.
      // expected-remark @below {{resolved '#pcf.ns_symtop' to !pcf.cluster<#pcf.test_scope, (0 -> d0), top>}}
      "test.dummy"() {test.resolve = "top"} : () -> ()
      // Resolve qualified path to outer named namespace.
      // expected-remark @below {{resolved '#pcf.ns_symouter.right' to !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>}}
      "test.dummy"() {test.resolve = "outer.right"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Same leaf name in multiple ancestor namespaces -- innermost wins.
// Both inner and outer anonymous namespaces define "a". Resolution
// from the inner namespace picks the inner "a".
util.func private @innermost_wins_same_leaf(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%a: !pcf.cluster<#pcf.test_scope, (0 -> d0), a>,
       %b: !pcf.cluster<#pcf.test_scope, (d0 -> s0), b>) {
    // expected-remark @below {{resolved '#pcf.ns_syma' to !pcf.cluster<#pcf.test_scope, (0 -> d0), a>}}
    pcf.shared_executor.tile_group %a split [[%j]]
        (%a2: !pcf.cluster<#pcf.test_scope, (0 -> d0), a>,
         %c: !pcf.cluster<#pcf.test_scope, (d0 -> s0), c>) {
      // "a" resolves to the inner namespace's definition.
      // expected-remark @below {{resolved '#pcf.ns_syma' to !pcf.cluster<#pcf.test_scope, (0 -> d0), a>}}
      "test.dummy"() {test.resolve = "a"} : () -> ()
      // "b" resolves to the outer namespace (not in inner).
      // expected-remark @below {{resolved '#pcf.ns_symb' to !pcf.cluster<#pcf.test_scope, (d0 -> s0), b>}}
      "test.dummy"() {test.resolve = "b"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), a>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}
