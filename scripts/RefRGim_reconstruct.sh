### 1=> 1KG reference panel path
### 2=> chromosome
### 3=> output_prefix
less $3.populations|cut -f 2|sort|uniq|xargs -i echo "gunzip -c $1/{}/chr$2.vcf.gz >$3.{}.$2.temp"|sh
gunzip -c $1/chr$2.info.gz > $3.chr$2.temp.info
less $3.populations|cut -f 3|sort|uniq > $3.$2.Sp.temp.info
for line in `cat $3.$2.Sp.temp.info`
do 
    if [ ! -d $3.$line  ];then
        mkdir -p $3.$line
    fi
    grep -w $line $3.populations|cut -f 2|sort|uniq|sed 's/$/.'$2'.temp/'|sed 's#^#'$3'.#'|awk 'BEGIN{RS="\n";ORS=" "}{print }'|xargs -i echo "paste $3.chr$2.temp.info {}"|sh |gzip >$3.$line/chr$2.vcf.gz
done    
echo "Reconstructed reference panel of chr$2 is finished ..."
rm -f ${3}.*.${2}.temp ${3}.${2}.Sp.temp.info ${3}.chr${2}.temp.info
