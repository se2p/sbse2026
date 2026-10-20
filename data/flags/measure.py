"""Compiles program.c with all 256 on/off combinations of 8 gcc optimisation
flags and records the resulting __TEXT segment size into sizes.csv.

This is macOS (clang standing in for /usr/bin/gcc), and the top-level __TEXT
segment size reported by `size` is padded to a 16KB page boundary and does
not change with the code at all. The actual code content lives in the
sections *inside* that segment (__text, __stubs, __const, __cstring,
__unwind_info, ...), whose combined size (the "total" line `size -m` prints
for the __TEXT segment) does move with the flags. That unpadded total is
what we use as the size metric.
"""
import subprocess, itertools, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "program.c")
OUT = os.path.join(HERE, "sizes.csv")
BINARY = "/tmp/flags_measure_bin"

FLAGS = [
    "inline-functions",
    "unroll-loops",
    "omit-frame-pointer",
    "tree-vectorize",
    "slp-vectorize",
    "stack-protector",
    "jump-tables",
    "wrapv",
]


def gcc_accepts(flag):
    r = subprocess.run(["/usr/bin/gcc", "-O2", flag, "-c", SRC, "-o", "/dev/null"],
                        capture_output=True, text=True)
    bad = ["not supported", "unknown argument", "error:"]
    return r.returncode == 0 and not any(b in r.stderr for b in bad)


def text_segment_size(binary):
    out = subprocess.run(["/usr/bin/size", "-m", binary],
                          capture_output=True, text=True).stdout
    in_text = False
    for line in out.splitlines():
        if line.startswith("Segment __TEXT:"):
            in_text = True
            continue
        if in_text:
            stripped = line.strip()
            if stripped.startswith("total"):
                return int(stripped.split()[-1])
            if line.startswith("Segment"):
                break
    raise RuntimeError("could not parse `size -m` output:\n" + out)


def compile_and_measure(config):
    flags = [f"-f{name}" if on else f"-fno-{name}" for name, on in zip(FLAGS, config)]
    r = subprocess.run(["/usr/bin/gcc", "-O2", *flags, SRC, "-o", BINARY],
                        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed for {flags}: {r.stderr}")
    size = text_segment_size(BINARY)
    os.remove(BINARY)
    return size


def main():
    for name in FLAGS:
        assert gcc_accepts(f"-f{name}"), f"-f{name} not accepted by /usr/bin/gcc"
        assert gcc_accepts(f"-fno-{name}"), f"-fno-{name} not accepted by /usr/bin/gcc"

    rows = []
    for config in itertools.product([0, 1], repeat=len(FLAGS)):
        size = compile_and_measure(config)
        rows.append(list(config) + [size])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"flag{i+1}" for i in range(len(FLAGS))] + ["size"])
        w.writerows(rows)

    sizes = [r[-1] for r in rows]
    print(f"flags: {FLAGS}")
    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"size range: {min(sizes)} - {max(sizes)} bytes")


if __name__ == "__main__":
    main()
