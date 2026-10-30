#!/bin/bash
# ArmAim2c RAM probe (14, 10-31 03:3x): bookkeeping on (60 its, crosses the 700 save) vs off (40 its), RSS every 15 s
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
O=logs/armaim2c/memprobe; mkdir -p $O
until [ -f logs/M3ROUTINE4_END.txt ] && [ logs/M3ROUTINE4_END.txt -nt $O/stamp ]; do sleep 30; done
trap 'echo "MEMPROBE_END $(date +%H:%M)" > $O/END.txt' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.5 AIB_FIX_STD=1 AIB_MEMDBG=1
read SW WTAG < logs/h29adopt/winner.txt
export AIB_SAIM_SWING=$(cd $(dirname $SW) && pwd -W)/$(basename $SW)
export AIB_SAIM_BOUND=$(grep -o "aim_bound [0-9.]*" logs/saim/progress.txt | tail -1 | awk '{print $2}')
run() {  # $1 tag $2 its
  idle
  $P -u scripts/train.py --task AIB-ArmAim2Comp-v0 --headless --num_envs 2048 --max_iterations $2 --run_name armaim2c_memprobe_$1 --resume True \
    --load_run 2026-10-31_02-20-04_armaim2c_2 --checkpoint model_650.pt > $O/train_$1.txt 2>&1 &
  TP=$!
  : > $O/rss_$1.txt
  while kill -0 $TP 2>/dev/null; do
    powershell.exe -NoProfile -Command "\$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'armaim2c_memprobe' }; foreach (\$x in \$p) { \$q = Get-Process -Id \$x.ProcessId; '{0} {1:N2} {2:N2}' -f (Get-Date -Format HH:mm:ss), (\$q.WorkingSet64/1GB), (\$q.PrivateMemorySize64/1GB) }" | tr -d '\r' >> $O/rss_$1.txt
    sleep 15
  done
}
echo "swing $AIB_SAIM_SWING bound $AIB_SAIM_BOUND" > $O/env.txt
run on 60
export AIB_ARMAIM_NOBOOK=1; run off 40
