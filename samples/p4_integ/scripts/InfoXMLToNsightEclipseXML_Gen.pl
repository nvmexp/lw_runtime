use File::Spec;
use File::Basename;
use File::Copy;
use XML::Simple;
use Cwd;
use Getopt::Long;

BEGIN {
    use File::Basename qw(dirname);
    unshift @INC, dirname(__FILE__);
}
use InfoHelper;

use strict;

my $numArgs = scalar @ARGV;
my $samples_list;
my $supported_sms;
my $samples_rootdir;

my $retval = GetOptions(
    "samples-list=s" => \$samples_list,
    "samples-rootdir=s" => \$samples_rootdir,
    "supported-sms=s" => \$supported_sms,
    "help|h" => sub { &Usage(); exit 0; },
);


my $list_file = $samples_list;
my $samples_root = $samples_rootdir;

if (!$samples_root)
{
    $samples_root = "../..";
}

if (!-e $list_file)
{
    print "InfoXMLToNsightEclipseXML_Gen.pl - Fatal error: Source file ($list_file) does not exist\n";
    exit(0);
}

if (!$supported_sms)
{
    Usage();
    exit(0);
}

my @samples = getSamplesList($list_file, $samples_root);

print "InfoXMLToNsightEclipseXML_Gen.pl - Obtained samples list...\n";

foreach my $sample (@samples)
{
    my $infoXmlPath = $sample . "/info.xml";
    if (!-e $infoXmlPath)
    {
        next;
    }

    print "InfoXMLToNsightEclipseXML_Gen.pl - Reading sample $infoXmlPath...\n";
    infoXmlToNsightEclipseXml($infoXmlPath, $sample);
}


sub Usage()
{
    print "Usage: \n";
    print "    CMD:  perl InfoXMLToNsightEclipseXML_Gen.pl --samples_list=samples_list.txt --samples-rootdir=[samples_root] --supported-sms=[SMs String]\n";
    print "    FUNC: Generates NsightEclipse.xml file for every info.xml found in samples_list.txt\n";
    print "          samples_list.txt should be one sample per line.\n";
    print "          samples_root defaults to ../.. and should point to the root of the samples branch\n";
    print "          supported_sms should be passed all the supported SMs in particular release given by lwconfig\n";
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
        print "InfoXMLToNsightEclipseXML_Gen.pl - Found sample $line\n";
    }
    
    return @samplesList;
}

sub infoXmlToNsightEclipseXml
{
    my $infoXml = shift @_;
    my $samplePath = shift @_;
    my $sampleXml = $infoXml;
    my $nsightEclipseXmlPath = $samplePath . "/NsightEclipse.xml";

    if (-e $infoXml)
    {     
        my @fileContents;
        my $name;
        my $description;
        my $minspec;
        my @keyconcepts;

        my $data = XMLin( $infoXml, ForceArray => 1);

        if ($data->{nsight_eclipse}[0] =~ m/false/i)
        {
            # Don't generate NsightEclipse.xml if the <nsight-eclipse> tag is false
            return;
        }

        my @inc_paths;
        if (scalar $data->{includepaths})
        {
            foreach (@{$data->{includepaths}[0]{path}})
            {
               push @inc_paths, $_;
            }
        }

        push @inc_paths, "./";
        push @inc_paths, "../";
        push @inc_paths, "../../../Common";

        @{$data->{includepaths}} = { path => [@inc_paths] };

        my $opengl = 0;
        for my $keyword (@{$data->{keywords}[0]{keyword}})
        {
            if ($keyword eq "openGL")
            {
                $opengl = 1;
            }
        }

        my @libraries_arr;
        if ($data->{libraries})
        {
            for my $library (@{$data->{libraries}[0]{library}})
            {
                if (ref($library) eq "HASH")
                {
                    next if ($library->{os} =~ m/windows/i);
                    push @libraries_arr, $library->{content};
                }
                else
                {
                    if ($library eq "lwca")
                    {
                        push @libraries_arr, {os => "linux", content =>"lwca"};
                        push @libraries_arr, {os => "macosx", framework => "true", content=>"LWCA"};
                    }
                    else
                    {
                        push @libraries_arr, $library;
                    }
                }
            }
        }

        my @library_paths;
        if ($data->{librarypaths})
        {
            for my $path (@{$data->{librarypaths}[0]{path}})
            {
                if (ref($path) eq "HASH")
                {
                    next if ($path->{os} =~ m/windows/i);
                    push @library_paths, $path->{content};
                }
                else
                {
                    push @library_paths, $path;
                }
            }
        }

        if ($opengl)
        {
            push @libraries_arr, "GLU";
            push @libraries_arr, "GL";
            push @libraries_arr, { os=> "macosx", framework => "true", content => "GLUT" };
            push @libraries_arr, { os=> "linux", content => "GLEW" };
            push @libraries_arr, { os=> "linux", content => "glut" };
            push @libraries_arr, { os=> "linux", content => "X11"};

            push @library_paths, {os => "linux", arch =>"x86_64", content => "../../../common/lib/linux/x86_64"};
            push @library_paths, {os => "linux", arch =>"armv7l", content => "../../../common/lib/linux/armv7l"};
            push @library_paths, {os => "macosx", content => "../../../common/lib/darwin"};
        }

        @{$data->{libraries}} = { library => [@libraries_arr] };
        @{$data->{librarypaths}} = { path => [@library_paths] };

        my $desktop_sms;
        my $tegra_sms;

        ($desktop_sms, $tegra_sms) = ParseSMsv2(@{$data->{supported_sm_architectures}}, $supported_sms);

        my @sms;
        @sms = uniq sort(@{$desktop_sms}, @{$tegra_sms});
        my $i=0;

        foreach my $sm (@sms)
        {
            $sms[$i++] = "sm$sm";
        }

        $data->{"sm-arch"} = [@sms];

        delete $data->{"owner"};
        delete $data->{"group"};
        delete $data->{"project_path"};
        delete $data->{"screenshots"};
        delete $data->{"screenshot"};
        delete $data->{"userguide"};
        delete $data->{"video"};
        delete $data->{"exelwtable"};
        delete $data->{"featured_date"};
        delete $data->{"supportedbuilds"};
        delete $data->{"gencode"};

        my $xml_nsight = XMLout($data, rootname => "entry", XMLDecl => '<?xml version="1.0" encoding="UTF-8"?> 
<!DOCTYPE entry SYSTEM "SamplesInfo.dtd">', NoEscape => 1 );

        # Have to do regex based CDATA addition as XML::Simple strips it out due to hashref used, also NO support in XMLout end to add CDATA.
        $xml_nsight =~ s/<description>/<description>\<!\[CDATA\[/ig;
        $xml_nsight =~ s/<\/description>/\]\]\><\/description>/ig;

        print "InfoXMLToNsightEclipseXML_Gen.pl - Writing NsightEclipse file: $nsightEclipseXmlPath\n";
        open NSIGHTECLIPSE, ">", $nsightEclipseXmlPath or print "InfoXMLToNsightEclipseXML_Gen.pl - Cannot write file: $!";

        print NSIGHTECLIPSE $xml_nsight; 

        close NSIGHTECLIPSE;        
    }
    else
    {
        print "InfoXMLToNsightEclipseXML_Gen.pl - Expected file does not exist: $infoXml\n";
    }
}
