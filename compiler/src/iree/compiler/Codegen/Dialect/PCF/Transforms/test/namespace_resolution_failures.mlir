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
