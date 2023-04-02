## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE VMVX benchmarks."""

from typing import List, Tuple
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import tflite_models
import benchmark_suites.iree.module_execution_configs
import benchmark_suites.iree.utils


class Android_VMVX_Benchmarks(object):
  """Benchmarks VMVX on Android devices."""

  VMVX_CPU_TARGET = iree_definitions.CompileTarget(
      target_backend=iree_definitions.TargetBackend.VMVX,
      target_architecture=common_definitions.DeviceArchitecture.VMVX_GENERIC,
      target_abi=iree_definitions.TargetABI.VMVX)
  DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_VMVX_GENERIC_DEFAULTS,
      tags=["default-flags"],
      compile_targets=[VMVX_CPU_TARGET])

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    gen_configs = [
        iree_definitions.ModuleGenerationConfig.with_flag_generation(
            compile_config=self.DEFAULT_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model)) for
        model in [tflite_models.MOBILENET_V2, tflite_models.MOBILENET_V3SMALL]
    ]
    default_execution_configs = [
        benchmark_suites.iree.module_execution_configs.
        get_vmvx_local_task_config(thread_num=4)
    ]
    big_cores_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        device_parameters={"big-cores"})
    run_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs,
        module_execution_configs=default_execution_configs,
        device_specs=big_cores_devices)

    return (gen_configs, run_configs)
