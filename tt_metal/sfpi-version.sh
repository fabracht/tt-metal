# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=b32c85972cc5f2c53036e589a3a8a3e4
sfpi_x86_64_Linux_deb_md5=ea7525ee0a719c719a23212b0e82d9e2
sfpi_x86_64_Linux_rpm_md5=fecba7019a067605276170fcf8cdb593
