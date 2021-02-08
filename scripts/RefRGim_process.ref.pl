#! /usr/bin/perl -w

use FindBin qw($Bin $Script);
use Getopt::Std;
use File::Path;
use File::Spec;
getopts('f:s:o:',\%opts);

die "perl $0
          STD  output
          -f   <input vcf file>
          -s   <sample info>
          -o   <chr>\n" unless ($opts{'f'} && $opts{'s'} && $opts{'o'});
my @sample;
my %uniq;
open INN,"$opts{'s'}" or die $!;
my $head=<INN>;
while(<INN>){
    chomp;
    my @arr=split(/\t/);
    push @sample,$arr[1];
    $uniq{$arr[1]}+=1;
    next if $uniq{$arr[1]}>1;
    my $file="./".$arr[1]."/chr".$opts{'o'}.".vcf.gz";
    open $arr[1],"|gzip >$file" or die $!;
}
close INN;
my %ind;
foreach(0..$#sample){
    push @{$ind{$sample[$_]}}, $_+9;
}
my $index;
my $out;
my $p1;
my $p2;
open INE, "|gzip >chr".$opts{'o'}.".info.gz" or die $!;
open IN,"gzip -dc $opts{'f'}|" or die $!;
foreach(1..252){
    my $zj=<IN>;
    print INE "$zj";
    foreach(keys %uniq){
        print $_ "\n";
    }
}
while (my $line=<IN>){
    chomp($line);
    my @arr=split(/\t/,$line);
    $p1=join("\t",@arr[0..8]);
    print INE "$p1\n";
    foreach(keys %uniq){
        $p2=join("\t",@arr[@{$ind{$_}}]);
        print $_ "$p2\n";
    }
}
close IN;
foreach(keys %uniq){
    close $_;
}
close INE;
print "Chr$opts{'o'} vcf file processing is done\n";
