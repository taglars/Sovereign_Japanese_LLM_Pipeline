#!/usr/bin/env bash
# What this account may do. Run on a login node.
#
#   bash cluster/doctor.sh
#
# `sinfo` is blocked for users on bwUniCluster; sinfo_t_idle replaces it.

echo "=============================================================="
echo "IDENTITY"
echo "=============================================================="
id
echo "login node: $(hostname)"
echo "home quota:"
( ws_list 2>/dev/null || true ) | head -5
quota -s 2>/dev/null | head -5 || echo "  (quota not reported here)"

echo
echo "=============================================================="
echo "SLURM ACCOUNT AND LIMITS"
echo "=============================================================="
echo "-- associations: which accounts/partitions/QOS you may submit to --"
sacctmgr -n show assoc user="$USER" \
    format=Account%20,Partition%16,QOS%20,MaxJobs,MaxSubmit 2>&1 | head -20
echo
echo "-- default account --"
sacctmgr -n show user "$USER" format=User%16,DefaultAccount%20 2>&1 | head -3
echo
echo "  If more than one Account is listed, every sbatch needs"
echo "  --account=<name> or it may be rejected or land on the wrong budget."

echo
echo "=============================================================="
echo "WHAT IS FREE RIGHT NOW"
echo "=============================================================="
echo "-- sinfo_t_idle (sinfo itself is restricted for users) --"
sinfo_t_idle 2>&1 | head -25

echo
echo "=============================================================="
echo "GPU PARTITIONS THIS PROJECT CARES ABOUT"
echo "=============================================================="
for part in gpu_h100 gpu_h100_short dev_gpu_h100 gpu_a100_il \
            gpu_a100_short gpu_h100_il dev_gpu_a100_il; do
    line=$(scontrol show partition "$part" 2>/dev/null \
        | tr '\n' ' ' \
        | grep -oE 'MaxTime=[^ ]+|TotalNodes=[^ ]+|State=[^ ]+|MaxNodes=[^ ]+' \
        | tr '\n' ' ')
    if [ -n "$line" ]; then
        printf '  %-16s %s\n' "$part" "$line"
    else
        printf '  %-16s (not visible to this account)\n' "$part"
    fi
done

echo
echo "=============================================================="
echo "YOUR JOBS"
echo "=============================================================="
squeue --me --format="%.10i %.14P %.16j %.8T %.10M %.10l %.6D %R" 2>&1 | head -20
echo
echo "-- recent history (last 3 days) --"
sacct --starttime now-3days \
    --format=JobID%14,JobName%16,Partition%14,State%16,Elapsed,MaxRSS,ExitCode \
    2>&1 | head -25

echo
echo "=============================================================="
echo "PROJECT STATE"
echo "=============================================================="
if command -v ws_find >/dev/null 2>&1; then
    WS=$(ws_find judge-validation 2>/dev/null || echo "")
    if [ -n "$WS" ]; then
        echo "workspace: $WS"
        ws_list 2>/dev/null | grep -A4 judge-validation | head -8
        echo
        echo "contents:"
        du -sh "$WS"/* 2>/dev/null | sed 's/^/  /' | head -10
    else
        echo "no workspace yet; cluster/00_setup.sh creates it"
    fi
fi
echo
echo "container: $([ -s "${JV_SQSH:-$HOME/nonexistent}" ] && echo present || echo "not imported")"
echo "PyYAML:    $(python3 -c 'import yaml; print("ok")' 2>/dev/null || echo MISSING)"
echo
echo "=============================================================="
