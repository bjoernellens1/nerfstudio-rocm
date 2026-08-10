Smoke test logs land in `logs/` (gitignored) when `scripts/smoke-test.sh` runs.
See `ROCM.md` §Phase 1 for the acceptance criteria these logs are checked against.

`splatfacto.log` — Splatfacto training on the bonsai mip-nerf360 scene
(7000 iterations, `images_4` downsample). See `ROCM.md` for the gsplat
verification this depends on.
