mode=${1:-turtle}
echo $mode
for i in  {1..5}; do
    grep " timesteps"  slurms/logs/all_pcgrl/pcgrl_smb_"$mode"_$i* | tail -n 1 | cut -d ":" -f 2 | cut -d ' ' -f 1 | xargs printf "%1.2e\n" 
done  #| awk 'BEGIN {T=0} {T+= $1} END {printf " Average Number of episodes for '$mode': " "%1.2e", T / NR}'