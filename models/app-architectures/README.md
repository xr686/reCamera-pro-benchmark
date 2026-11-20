# RV1126B classification checkpoints

These FP16 RKNN binaries were used for the on-device benchmarks. Each linked timm model card identifies the upstream checkpoint, credits and Apache-2.0 license. Conversion uses fixed inputs and RKNN-Toolkit2 2.3.2. SwiftFormer and VOLO use additional [graph rewrites](../../apps/npu-benchmark/ARCHITECTURES-1.3.2.md).

| Architecture | RKNN | Upstream model card |
|---|---|---|
| GhostNet | [ghostnet](ghostnet.rknn) | [ghostnet_100.in1k](https://huggingface.co/timm/ghostnet_100.in1k/blob/e524c2ee5d5a9412f802ce38ff895aba07b30910/README.md) |
| MixNet | [mixnet-s](mixnet-s.rknn) | [mixnet_s.ft_in1k](https://huggingface.co/timm/mixnet_s.ft_in1k/blob/94149f6f14da590e98eeaa4dd9867b97a23bd3f4/README.md) |
| PP-LCNet | [lcnet](lcnet.rknn) | [lcnet_100.ra2_in1k](https://huggingface.co/timm/lcnet_100.ra2_in1k/blob/3a2ceedadb0ace67bcbfb840a64caeb044dc8008/README.md) |
| DPN | [dpn68](dpn68.rknn) | [dpn68.mx_in1k](https://huggingface.co/timm/dpn68.mx_in1k/blob/e652eb27a553819f870f33ddd3ee03f6f701ecd8/README.md) |
| EfficientFormer | [efficientformer](efficientformer.rknn) | [efficientformer_l1.snap_dist_in1k](https://huggingface.co/timm/efficientformer_l1.snap_dist_in1k/blob/8738a58190800aaf5995f453b05f643df453a565/README.md) |
| PoolFormer | [poolformer](poolformer.rknn) | [poolformer_s12.sail_in1k](https://huggingface.co/timm/poolformer_s12.sail_in1k/blob/7f922b14752901f7a3c6be4a9c2fc058dc4f1a12/README.md) |
| VoVNet | [vovnet](vovnet.rknn) | [ese_vovnet19b_dw.ra_in1k](https://huggingface.co/timm/ese_vovnet19b_dw.ra_in1k/blob/39808ce845d469f0f39299fdde40a2ef4a406268/README.md) |
| LeViT | [levit](levit.rknn) | [levit_128s.fb_dist_in1k](https://huggingface.co/timm/levit_128s.fb_dist_in1k/blob/ff86cd4479c83f857075f614ddf7e093dbf1292f/README.md) |
| CoaT | [coat-lite](coat-lite.rknn) | [coat_lite_tiny.in1k](https://huggingface.co/timm/coat_lite_tiny.in1k/blob/db143d32976054b06287b8687ff55b9d443fb82f/README.md) |
| FasterNet | [fasternet-t0](fasternet-t0.rknn) | [fasternet_t0.in1k](https://huggingface.co/timm/fasternet_t0.in1k/blob/a8c465730c7896ba51d860a6e384fad94e7cfb44/README.md) |
| PVT | [pvtv2-b0](pvtv2-b0.rknn) | [pvt_v2_b0.in1k](https://huggingface.co/timm/pvt_v2_b0.in1k/blob/05b6ee72c4090dfd6e0b95a73db6e20cd1e75e9f/README.md) |
| GERNet | [gernet-s](gernet-s.rknn) | [gernet_s.idstcv_in1k](https://huggingface.co/timm/gernet_s.idstcv_in1k/blob/f07782a90ed90f4108204acbcd8cd0c156419b97/README.md) |
| FBNet | [fbnetc](fbnetc.rknn) | [fbnetc_100.rmsp_in1k](https://huggingface.co/timm/fbnetc_100.rmsp_in1k/blob/d48d5c1d41488815cf985be3aebeb2641f5ca280/README.md) |
| HardCoRe-NAS | [hardcorenas-a](hardcorenas-a.rknn) | [hardcorenas_a.miil_green_in1k](https://huggingface.co/timm/hardcorenas_a.miil_green_in1k/blob/8043f94dc3f54de7662abf53696ce29a9bb811be/README.md) |
| StarNet | [starnet](starnet.rknn) | [starnet_s1.in1k](https://huggingface.co/timm/starnet_s1.in1k/blob/e3fbe6d1e24b6293fa6e7b897f380bd62f77e01f/README.md) |
| SwiftFormer | [swiftformer](swiftformer.rknn) | [swiftformer_xs.dist_in1k](https://huggingface.co/timm/swiftformer_xs.dist_in1k/blob/ac0196f198e58c82183a67f5f5c0952421b3e6ac/README.md) |
| PP-HGNet | [hgnet](hgnet.rknn) | [hgnet_tiny.ssld_in1k](https://huggingface.co/timm/hgnet_tiny.ssld_in1k/blob/60fd3a062040c894f9dd75dfc1887843b71ab8ef/README.md) |
| XCiT | [xcit](xcit.rknn) | [xcit_nano_12_p16_224.fb_dist_in1k](https://huggingface.co/timm/xcit_nano_12_p16_224.fb_dist_in1k/blob/621eefa447a264fd3ea0d55ae771f7ba84c58727/README.md) |
| Xception | [xception](xception.rknn) | [xception41p.ra3_in1k](https://huggingface.co/timm/xception41p.ra3_in1k/blob/39c1434e60cd96017b21dd204b26df2716f24136/README.md) |
| DaViT | [davit](davit.rknn) | [davit_tiny.msft_in1k](https://huggingface.co/timm/davit_tiny.msft_in1k/blob/982f32c73d98f9966df4a7178b32b820bed3c1b1/README.md) |
| gMLP | [gmlp](gmlp.rknn) | [gmlp_s16_224.ra3_in1k](https://huggingface.co/timm/gmlp_s16_224.ra3_in1k/blob/ef4294efcbaed5fb374565716d9e08e4ed69a6d8/README.md) |
| TNT | [tnt](tnt.rknn) | [tnt_s_patch16_224.in1k](https://huggingface.co/timm/tnt_s_patch16_224.in1k/blob/3197ffd33ba33a86ddfe3719fd6ab87a479ff6e8/README.md) |
| MaxViT | [maxvit](maxvit.rknn) | [maxvit_tiny_tf_224.in1k](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k/blob/041f2cce4d74c7539d63aa9fb85786e78072d487/README.md) |
| VOLO | [volo](volo.rknn) | [volo_d1_224.sail_in1k](https://huggingface.co/timm/volo_d1_224.sail_in1k/blob/0d1b31899ae4cb562932c62d1d8820c1d796a4f1/README.md) |

The [source index](../../benchmark_results/app/classification-sources.csv) covers all tested classification checkpoints. The [artifact manifest](../../benchmark_results/app/model-hashes.csv) records the SHA-256 for all 270 measured configurations.
