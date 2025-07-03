use File::Spec;
use File::Basename;
use File::Copy;
use XML::Simple;
use Getopt::Long;
use Cwd;

BEGIN {
    use File::Basename qw(dirname);
    unshift @INC, dirname(__FILE__);
}
use InfoHelper;

use strict;

my $numArgs = scalar @ARGV;
my $samples_list;
my $supported_sms;
my @supported_sms_arr;
my $samples_root;

# check arguments
my $retval = GetOptions(
    "samples-list=s" => \$samples_list,
    "samples-rootdir=s" => \$samples_root,
    "supported-sms=s" => \$supported_sms,
    "help|h" => sub { &Usage(); exit 0; },
);

exit 1 if ($retval != 1);

if (!-e $samples_list)
{
    print "InfoXMLToReadme_Gen.pl - Fatal error: Source file ($samples_list) does not exist\n";
    exit(0);
}

if (!$supported_sms)
{
    Usage();
    exit(0);
}
else 
{
    if (!$samples_root)
    {
        $samples_root = "../..";
    }

    my @samples = getSamplesList($samples_list, $samples_root);

    # This is to replace "20 30 32 35 37 50 52 53" to be "2.0 3.0 3.2 3.5 3.7 5.0 5.2 5.3"
    my @treated_sms_arr = split / /,$supported_sms;
    foreach (@treated_sms_arr)
    {
        substr($_, -1, 0, ".");
        push(@supported_sms_arr, $_);
    }
    
    print "InfoXMLToReadme_Gen.pl - Obtained project list...\n";

    foreach my $sample (@samples)
    {
        my $infoXmlPath = $sample . "/info.xml";
        if (!-e $infoXmlPath)
        {   
            next;
    	}
        
        print "InfoXMLToReadme_Gen.pl - Reading project $infoXmlPath...\n";
        infoXmlToReadme($infoXmlPath, $sample);
    }
}

sub Usage
{
    print "Usage: \n";
    print "    CMD:  perl InfoXMLToReadme_Gen.pl --samples_list=samples_list.txt --samples-rootdir=[samples_root] --supported-sms=[SMs String]\n";
    print "    FUNC: Generates readme.txt file for every info.xml found in projects\n";
    print "          samples_list.txt should be one sample per line.\n";
    print "          samples_root defaults to ../.. and should point to the root of the samples branch\n";
    print "          supported_sms should be passed all the supported SMs in particular release given by lwconfig\n";
    exit(0);    
}

sub getSamplesList
{
    my $listPath = shift @_;
    my $root = shift @_;
    my @fileContents;
    
    open(LISTFILE, $listPath);
    while (<LISTFILE>) { push @fileContents, $_; }
    close(LISTFILE);
    
    my @samplesList;
    foreach my $line (@fileContents)
    {
        chomp $line;
        next if ($line =~ m/^\s*#/);

        $line = "$root/$line";
        $line =~ s|/+|/|g;
        push @samplesList, $line;
        print "InfoXMLToReadme_Gen.pl - Found sample $line\n";
    }
    
    return @samplesList;
}

sub infoXmlToReadme
{
    my $infoXml = shift @_;
    my $samplePath = shift @_;
    my $sampleXml = $infoXml;
    my $readmePath = $samplePath . "/readme.txt";

    if (-e $infoXml)
    {
        my @fileContents;
        my $name;
        my $description;
        my $minspec = $supported_sms_arr[0];

        my $data = XMLin(
        $sampleXml,
        ForceArray => [
            "concept",
            ],
        );

        $name = $data->{"name"};

        if (length $data->{"supported_sm_architectures"}->{"from"})
        {
            $minspec = $data->{"supported_sm_architectures"}->{"from"};
        }
        elsif (length $data->{"supported_sm_architectures"}->{"include"})
        {
            if ($data->{"supported_sm_architectures"}->{"include"} != "all")
            {
                $minspec = $data->{"supported_sm_architectures"}->{"include"};
            }
        }

        $description = $data->{"description"};
        # Process description
        $description =~ s/\<!\[CDATA\[//ig;
        $description =~ s/\]\]\>//ig;

        my $outputText = "Sample: $name\n";
        $outputText .= "Minimum spec: SM $minspec\n\n";
        $outputText .= "$description\n\n";

        if (scalar $data->{keyconcepts})
        {
            $outputText .= "Key concepts:\n";
        }

        for my $concept (@{$data->{keyconcepts}->{concept}})
        {
            if (ref($concept) eq "HASH" && $concept->{level})
            {
                $outputText .= "$concept->{content}\n";
            }
            else
            {
                $outputText .= "$concept\n";
            }
        }

        print "InfoXMLToReadme_Gen.pl - Writing readme file: $readmePath\n";
        open README, ">", $readmePath or print "InfoXMLToReadme_Gen.pl - Cannot write file: $!";
        print README $outputText;
        close README;
    }
    else
    {
        print "InfoXMLToReadme_Gen.pl - Expected file does not exist: $infoXml\n";
    }
}
