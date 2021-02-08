less integrated_call_samples_v3.20130502.ALL.panel|sed '1d'|cut -f 2|sort|uniq|xargs -i echo "mkdir -p {}"|sh
for((i=1;i<=22;i++));  
do   
    while [ 1 ]; do
        num=`ps -ef|grep RefRGim_process.ref.pl|wc -l`
        if [ $num -lt 6 ]; then
            perl ../scripts/RefRGim_process.ref.pl -f ALL.chr${i}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz -s integrated_call_samples_v3.20130502.ALL.panel -o $i &
            break
        fi
        sleep 100s
    done
done  

while [ 1 ]; do
    num=`ps -ef|grep RefRGim_process.ref.pl|wc -l`
    if [ $num -eq 1 ]; then
        echo "References panel processing is done."
        break
    fi
done
