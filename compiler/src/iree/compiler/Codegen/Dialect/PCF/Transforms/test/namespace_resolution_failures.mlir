// RUN: iree-opt --iree-pcf-test-namespace-resolution --allow-unregistered-dialect --verify-diagnostics --split-input-file %s

// Resolution failure: nonexistent namespace.
util.func private @resolution_failure_no_ns(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), tg.right>) {
    // expected-error @+1 {{no namespace named 'nonexistent' found in ancestor chain}}
    "test.dummy"() {test.resolve = "nonexistent.foo"} : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Resolution failure: leaf symbol not found.
util.func private @resolution_failure_no_leaf(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // expected-error @+1 {{failed to resolve leaf symbol 'missing'}}
    "test.dummy"() {test.resolve = "missing"} : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Resolution failure: no enclosing namespace.
// expected-error @+1 {{no enclosing namespace found for symbol}}
"test.dummy"() {test.resolve = "anything"} : () -> ()

// -----

// Resolution failure: qualified path where namespace exists but leaf doesn't.
util.func private @resolution_failure_ns_exists_no_leaf(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), tg.right>) {
    // expected-error @+1 {{symbol 'missing' not found in namespace}}
    "test.dummy"() {test.resolve = "tg.missing"} : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Resolution failure: qualified path with intermediate anonymous namespace
// blocking contiguous match. The path "outer.anon.top" requires "anon" to be
// a named namespace contiguous to "outer", but it's anonymous.
util.func private @resolution_failure_anon_blocks_path(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    // expected-remark @below {{resolved '#pcf.ns_symouter.left' to !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>}}
    pcf.shared_executor.tile_group %left split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), bot>) {
      // "outer.anon.top" -- "anon" is not a named namespace, should fail.
      // expected-error @+1 {{expected namespace 'anon' at ancestor position but found anonymous namespace}}
      "test.dummy"() {test.resolve = "outer.anon.top"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Resolution failure: qualified path targeting a sibling namespace.
// Two tile_groups at the same level with different named namespaces.
// Resolving "sibling.x" from inside "here" should fail because resolution
// only walks upward, never sideways.
util.func private @resolution_failure_sibling_namespace(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // expected-remark @below {{resolved '#pcf.ns_symleft' to !pcf.cluster<#pcf.test_scope, (0 -> d0), left>}}
    pcf.shared_executor.tile_group %left ns(here) split [[%j]]
        (%a: !pcf.cluster<#pcf.test_scope, (0 -> d0), here.a>,
         %b: !pcf.cluster<#pcf.test_scope, (d0 -> s0), here.b>) {
      // Trying to resolve "sibling.x" -- no namespace named "sibling" in
      // ancestor chain.
      // expected-error @+1 {{no namespace named 'sibling' found in ancestor chain}}
      "test.dummy"() {test.resolve = "sibling.x"} : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}
