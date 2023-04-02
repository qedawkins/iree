# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""List of unique random IDs in the framework and id utilities.

Each ID should be generated from uuid.uuid4().
"""

import hashlib
from typing import Sequence

# Special id which will be ignored when calculating the composite id.
#
# It should only be used when adding a new field to a composite object while
# we want to maintain the same id on the existing composite objects.
#
# In such case, you need to create a "default config" for the new field with
# this id and populate that config to the fields of the existing objects. The
# composite id computing function will ignore this id and keep the output
# unchanged.
TRANSPARENT_ID = "00000000-0000-0000-0000-000000000000"


def hash_composite_id(keys: Sequence[str]) -> str:
  """Computes the composite hash id from string keys.

  String keys are the component ids that compose this composite object. We hash
  the composite id since the id isn't designed to be inspected and insufficient
  to reconstruct the original composite object.

  Note that the output is sensitive to the order of the keys, and any key ==
  TRANSPARENT_ID will be skipped. When adding a new key to the keys, the new key
  should be always appended to the end. In this way, the composite id can be
  unchanged for the existing composite object if they use TRANSPARENT_ID on the
  new keyed field.

  The composite id is computed in the following steps:
  1. Index each key with its position in the list from 0.
  2. Remove any key == TRANSPARENT_ID
  3. Get the SHA256 hex digest of "0-key_0:1-key_1:..."

  Step 1 is needed to avoid the ambiguity between:
  ["key_abc", TRANSPARENT_ID] and [TRANSPARENT_ID, "key_abc"]
  since after removing TRANSPARENT_ID, they both become ["key_abc"] without the
  position index.

  Args:
    keys: list of string keys.

  Returns:
    Unique composite id.
  """
  trimmed_indexed_key = [
      f"{index}-{key}" for index, key in enumerate(keys)
      if key != TRANSPARENT_ID
  ]
  return hashlib.sha256(
      ":".join(trimmed_indexed_key).encode("utf-8")).hexdigest()

# To generate an id, run `uuid.uuid4()`.

# Models.
#    TFLite.
MODEL_DEEPLABV3_FP32 = "c36c63b0-220a-4d78-8ade-c45ce47d89d3"
MODEL_MOBILESSD_FP32 = "0e466f69-91d6-4e50-b62b-a82b6213a231"
MODEL_POSENET_FP32 = "5afc3014-d29d-4e88-a840-fbaf678acf2b"
MODEL_MOBILEBERT_FP32 = "cc69d69f-6d1f-4a1a-a31e-e021888d0d28"
MODEL_MOBILEBERT_INT8 = "e3997104-a3d2-46b4-9fbf-39069906d123"
MODEL_MOBILEBERT_FP16 = "73a0402e-271b-4aa8-a6a5-ac05839ca569"
MODEL_MOBILENET_V1 = "78eab9e5-9ff1-4769-9b55-933c81cc9a0f"
MODEL_MOBILENET_V2 = "7d45f8e5-bb5e-48d0-928d-8f125104578f"
MODEL_MOBILENET_V3SMALL = "58855e40-eba9-4a71-b878-6b35e3460244"
MODEL_PERSON_DETECT_INT8 = "bc1338be-e3df-44fd-82e4-40ba9560a073"
MODEL_EFFICIENTNET_INT8 = "4a6f545e-1b4e-41a5-9236-792aa578184b"
#    Tensorflow.
MODEL_MINILM_L12_H384_UNCASED_INT32_SEQLEN128 = "ecf5c970-ee97-49f0-a4ed-df1f34e9d493"
MODEL_BERT_FOR_MASKED_LM_FP32_SEQLEN512_TF = "39d157ad-f0ec-4a76-963b-d783beaed60f"
MODEL_EFFICIENTNET_V2_S_FP32_TF = "ebe7897f-5613-435b-a330-3cb967704e5e"
MODEL_RESNET50_TF_FP32 = "c393b4fa-beb4-45d5-982a-c6328aa05d08"
MODEL_BERT_LARGE_TF_FP32_SEQLEN384 = "8871f602-571c-4eb8-b94d-554cc8ceec5a"
#    PyTorch.
MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH = "9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8"
MODEL_UNET_2D_FP32_TORCH = "340553d1-e6fe-41b6-b2c7-687c74ccec56"

# Model input data
MODEL_INPUT_DATA_ZEROS = "8d4a034e-944d-4725-8402-d6f6e61be93c"

# Devices
DEVICE_SPEC_GCP_C2_STANDARD_16 = "9a4804f1-b1b9-46cd-b251-7f16a655f782"
DEVICE_SPEC_GCP_A2_HIGHGPU_1G = "78c56b95-2d7d-44b5-b5fd-8e47aa961108"
DEVICE_SPEC_MOBILE_PIXEL_4 = "fc901efc-ddf8-44c0-b009-8eecb8286521"
DEVICE_SPEC_MOBILE_PIXEL_6_PRO = "0f624778-dc50-43eb-8060-7f4aea9634e1"
DEVICE_SPEC_MOBILE_MOTO_EDGE_X30 = "dae12e6a-4169-419d-adc9-4f63a2ff1997"
DEVICE_SPEC_MOBILE_SM_G980F = "0247f709-a300-4d12-b986-bdb0c15c2653"

# IREE benchmarks
IREE_COMPILE_CONFIG_VMVX_GENERIC_DEFAULTS = "75336abd-8108-462c-9ce3-15443e3f32f4"
IREE_COMPILE_CONFIG_LINUX_CASCADELAKE = "e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_FUSE_PADDING = "6d0d5716-5525-44ad-b71d-8075ee1583a6"
IREE_COMPILE_CONFIG_LINUX_RV64_GENERIC_DEFAULTS = "cdf579a9-5446-403b-a991-802a6c702e65"
IREE_COMPILE_CONFIG_LINUX_RV32_GENERIC_DEFAULTS = "6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4"
IREE_COMPILE_CONFIG_LINUX_CUDA_SM80_DEFAULTS = "09cb5300-7f73-45cf-9f68-e114c77ca030"
IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_DEFAULTS = "8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_FUSE_PADDING = "32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
IREE_COMPILE_CONFIG_ANDROID_VALHALL_MALI_FUSE_PADDING_REPEATED_KERNEL = "6b601a8d-4824-42e0-bcc6-500c0c3fa346"
IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_DEFAULTS = "1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D = "d463322c-24e6-4685-85ca-d541b41a405f"
IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D_DOTPROD = "f672a6b9-99fc-47ce-8b1b-8e5f44a541a1"
IREE_COMPILE_CONFIG_ANDROID_ADRENO_GENERIC_DEFAULTS = "c7eea358-d8d2-4199-9d75-bb741c399b1b"
IREE_COMPILE_CONFIG_ANDROID_ADRENO_GENERIC_FUSE_PADDING = "d3038b95-c889-456a-bff6-5cbabd10f1ad"
IREE_COMPILE_CONFIG_ANDROID_ADRENO_GENERIC_FUSE_PADDING_REPEATED_KERNEL = "70b823ca-2807-4531-8c00-e02af7d70466"
IREE_MODULE_EXECUTION_CONFIG_LOCAL_SYNC = "13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03"
IREE_MODULE_EXECUTION_CONFIG_LOCAL_TASK_BASE = "c7c4a15e-b20c-4898-bb4a-864f34ff34b2"
IREE_MODULE_EXECUTION_CONFIG_CUDA = "f7c0ec98-f028-436a-b05a-7d35cf18ce2d"
IREE_MODULE_EXECUTION_CONFIG_VULKAN = "34ae13f0-d6d9-43f7-befb-15d024e88e89"
IREE_MODULE_EXECUTION_CONFIG_VULKAN_BATCH_SIZE_16 = "b10737a8-5da4-4052-9b7a-5b07f21e02d0"
IREE_MODULE_EXECUTION_CONFIG_VULKAN_BATCH_SIZE_32 = "c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8"
IREE_MODULE_EXECUTION_CONFIG_VMVX_LOCAL_TASK_BASE = "953183e2-1e84-4a51-a43c-9b869bdc2218"
IREE_MODEL_IMPORT_TF_V1_DEFAULT = "8b2df698-f3ba-4207-8696-6c909776eac4"
IREE_MODEL_IMPORT_TF_V2_DEFAULT = "2464f30a-d04c-4f46-a332-b8c7853d61fc"
IREE_MODEL_IMPORT_TFLITE_DEFAULT = "16280d67-7ce0-4807-ab4b-0cb3c771d206"
IREE_MODEL_IMPORT_LINALG_MLIR_DEFAULT = "8afc4561-e84d-4a91-af55-2b1917465fcc"
