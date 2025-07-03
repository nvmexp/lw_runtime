package InfoHelper;

use strict;
use Data::Dumper;
use XML::Simple;
use File::Basename;
use vars qw($VERSION @ISA @EXPORT);

@ISA = qw(Exporter);

@EXPORT = qw(
ParseSMs
GetSMs
ParseSMsv2
uniq
);

$VERSION = '1.0';

use XML::Simple;

my @attrs = (NoAttr => 1, KeepRoot => 1, NoSort => 1);

my @default_desktop_sms = ("30", "35", "37", "50", "52", "60", "61", "70");
my @default_tegra_sms = ("30", "32", "35", "37", "50", "52", "53", "60", "61", "62", "70", "72");
my @tegra_only_sms = ("32", "53", "62", "72");

sub trim
{
    my $text = shift;

    $text =~ s/^\s*//;
    $text =~ s/\s*$//;

    return $text;
}

sub uniq
{
    my %seen = ();
    my @ret = ();

    foreach my $element (@_)
    {
        unless ($seen{$element})
        {
            push @ret, $element;
            $seen{$element} = 1;
        }
    }

    return @ret;
}

sub GetSMs
{
    return (\@default_desktop_sms, \@default_tegra_sms);
}

sub ParseSMs
{
    my $sm_data = shift;
    my $all_supported_sms = shift;
    my @supported_desktop_sms;
    my @supported_tegra_sms;

    if (defined $all_supported_sms)
    {
        @supported_desktop_sms = split(' ', $all_supported_sms);
        my $exists = 0;
        my $j = 0;
        for my $i (0 .. $#supported_desktop_sms)
        {
            foreach (@tegra_only_sms)
            {
                if ($supported_desktop_sms[$i] eq $_)
                {
                    $exists = 1;
                }
            }
            if ($exists eq 0)
            {
                $supported_desktop_sms[$j++] = $supported_desktop_sms[$i];
            }
            $exists = 0;
        }
        @supported_desktop_sms = @supported_desktop_sms[0 .. $j-1];
        @supported_tegra_sms   = split(' ', $all_supported_sms);
    }
    else
    {
        @supported_desktop_sms = @default_desktop_sms;
        @supported_tegra_sms   = @default_tegra_sms;
    }

    my @tegra_sms;
    my @desktop_sms;

    my @exclude_desktop_sms;
    my @exclude_tegra_sms;
    my $exclude_all_desktop = 0;
    my $exclude_all_tegra = 0;

    for (@{$sm_data->{exclude}})
    {
        my $content;

        if (ref($_) eq "HASH")
        {
            $content = $_->{content};
            
            if ($_->{type} =~ m/desktop/i)
            {
                if ($content ne "all")
                {
                    push (@exclude_desktop_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_desktop_sms)
                    {
                        push (@exclude_desktop_sms, $def_sm);
                    }
                    $exclude_all_desktop = 1;
                }
            }
            elsif ($_->{type} =~ m/cheetah/i)
            {
                if ($content ne "all")
                {
                    push (@exclude_tegra_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_tegra_sms)
                    {
                        push (@exclude_tegra_sms, $def_sm);
                    }
                    $exclude_all_tegra = 1;
                }
            }
        }
        else
        {
            $content = $_;
            if ($content ne "all")
            {
                push (@exclude_desktop_sms, $content);
                push (@exclude_tegra_sms, $content);
            }
            else
            {
                for my $def_sm (@supported_desktop_sms)
                {
                    push (@exclude_desktop_sms, $def_sm);
                }

                for my $def_sm (@supported_tegra_sms)
                {
                    push (@exclude_tegra_sms, $def_sm);
                }

                $exclude_all_desktop = 1;
                $exclude_all_tegra = 1;
            }
        }
    }

    my @include_desktop_sms;
    my @include_tegra_sms;
    my $include_all_desktop = 0;
    my $include_all_tegra = 0;

    for (@{$sm_data->{include}})
    {
        my $content;

        if (ref($_) eq "HASH")
        {
            $content = $_->{content};
            
            if ($_->{type} =~ m/desktop/i)
            {
                if ($content ne "all")
                {
                    push (@include_desktop_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_desktop_sms)
                    {
                        push (@include_desktop_sms, $def_sm);
                    }
                    $include_all_desktop = 1;
                }
            }
            elsif ($_->{type} =~ m/cheetah/i)
            {
                if ($content ne "all")
                {
                    push (@include_tegra_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_tegra_sms)
                    {
                        push (@include_tegra_sms, $def_sm);
                    }
                    $include_all_tegra = 1;
                }
            }
        }
        else
        {
            $content = $_;
            if ($content ne "all")
            {
                push (@include_desktop_sms, $content);
                push (@include_tegra_sms, $content);
            }
            else
            {
                for my $def_sm (@supported_desktop_sms)
                {
                    push (@include_desktop_sms, $def_sm);
                }

                for my $def_sm (@supported_tegra_sms)
                {
                    push (@include_tegra_sms, $def_sm);
                }

                $include_all_desktop = 1;
                $include_all_tegra = 1;
            }
        }
    }

    @include_desktop_sms = UniqArray(@include_desktop_sms);
    @include_tegra_sms = UniqArray(@include_tegra_sms);
    @exclude_desktop_sms = UniqArray(@exclude_desktop_sms);
    @exclude_tegra_sms = UniqArray(@exclude_tegra_sms);

    die "Must have only all SMs included, or only all SMs excluded." unless ($include_all_desktop ^ $exclude_all_desktop);
    die "Must have only all SMs included, or only all SMs excluded." unless ($include_all_tegra ^ $exclude_all_tegra);

    for my $sm (@include_desktop_sms)
    {
        push (@desktop_sms, $sm);
    }

    if ($include_all_desktop)
    {
        for (@exclude_desktop_sms)
        {
            my $count = scalar @desktop_sms;
            my $index = 0;
            $index++ until @desktop_sms[$index] eq $_ or $index >= $count;
            splice(@desktop_sms, $index, 1);
        }
    }

    for my $sm (@include_tegra_sms)
    {
        push (@tegra_sms, $sm);
    }

    if ($include_all_tegra)
    {
        for (@exclude_tegra_sms)
        {
            my $count = scalar @tegra_sms;
            my $index = 0;
            $index++ until @tegra_sms[$index] eq $_ or $index >= $count;
            splice(@tegra_sms, $index, 1);
        }
    }

    return (\@desktop_sms, \@tegra_sms);
}

# ParseSMsv2 provides a interface for generating supported SM values from info. in <supported_sm_architecture> tag.
# This is lwrrently only used for NsightEclipse.xml generation
sub ParseSMsv2
{
    my $sm_data =  shift;
    my $all_supported_sms = shift;
    my @supported_desktop_sms;
    my @supported_tegra_sms;

    if (defined $all_supported_sms)
    {
        @supported_desktop_sms = split(' ', $all_supported_sms);
        my $exists = 0;
        my $j = 0;
        for my $i (0 .. $#supported_desktop_sms)
        {
            foreach (@tegra_only_sms)
            {
                if ($supported_desktop_sms[$i] eq $_)
                {
                    $exists = 1;
                }
            }
            if ($exists eq 0)
            {
                $supported_desktop_sms[$j++] = $supported_desktop_sms[$i];
            }
            $exists = 0;
        }
        @supported_desktop_sms = @supported_desktop_sms[0 .. $j-1];
        @supported_tegra_sms   = split(' ', $all_supported_sms);
    }
    else
    {
        print "supported-sms not passed. exiting\n ";
        exit(0);
    }

    my @tegra_sms;
    my @desktop_sms;

    my @exclude_desktop_sms;
    my @exclude_tegra_sms;
    my $exclude_all_desktop = 0;
    my $exclude_all_tegra = 0;

    for (@{$sm_data->{exclude}})
    {
        my $content;

        if (ref($_) eq "HASH")
        {
            $content = $_->{content};
            $content =~ s/^\s+|\s+$//g;
            $content =~ s/\.//g; 

            if ($_->{type} =~ m/desktop/i)
            {
                if ($content ne "all")
                {
                    push (@exclude_desktop_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_desktop_sms)
                    {
                        push (@exclude_desktop_sms, $def_sm);
                    }
                    $exclude_all_desktop = 1;
                }
            }
            elsif ($_->{type} =~ m/cheetah/i)
            {
                if ($content ne "all")
                {
                    push (@exclude_tegra_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_tegra_sms)
                    {
                        $def_sm =~ s/^\s+|\s+$//g;
                        $def_sm =~ s/\.//g; 
                        push (@exclude_tegra_sms, $def_sm);
                    }
                    $exclude_all_tegra = 1;
                }
            }
        }
        else
        {
            $content = $_;

            if ($content ne "all")
            {
                $content =~ s/^\s+|\s+$//g;
                $content =~ s/\.//g;
                push (@exclude_desktop_sms, $content);
                push (@exclude_tegra_sms, $content);
            }
            else
            {
                for my $def_sm (@supported_desktop_sms)
                {
                    push (@exclude_desktop_sms, $def_sm);
                }

                for my $def_sm (@supported_tegra_sms)
                {
                    push (@exclude_tegra_sms, $def_sm);
                }

                $exclude_all_desktop = 1;
                $exclude_all_tegra = 1;
            }
        }
    }

    my @include_desktop_sms;
    my @include_tegra_sms;
    my $include_all_desktop = 0;
    my $include_all_tegra = 0;

    for (@{$sm_data->{include}})
    {
        my $content;

        if (ref($_) eq "HASH")
        {
            $content = $_->{content};
            
            if ($_->{type} =~ m/desktop/i)
            {
                if ($content ne "all")
                {
                    $content =~ s/^\s+|\s+$//g;
                    $content =~ s/\.//g; 
                    push (@include_desktop_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_desktop_sms)
                    {
                        push (@include_desktop_sms, $def_sm);
                    }
                    $include_all_desktop = 1;
                }
            }
            elsif ($_->{type} =~ m/cheetah/i)
            {
                if ($content ne "all")
                {
                    $content =~ s/^\s+|\s+$//g;
                    $content =~ s/\.//g; 
                    push (@include_tegra_sms, $content);
                }
                else
                {
                    for my $def_sm (@supported_tegra_sms)
                    {
                        push (@include_tegra_sms, $def_sm);
                    }
                    $include_all_tegra = 1;
                }
            }
        }
        else
        {
            $content = $_;
            if ($content ne "all")
            {
                $content =~ s/^\s+|\s+$//g;
                $content =~ s/\.//g; 
                push (@include_desktop_sms, $content);
                push (@include_tegra_sms, $content);
            }
            else
            {
                for my $def_sm (@supported_desktop_sms)
                {
                    push (@include_desktop_sms, $def_sm);
                }

                for my $def_sm (@supported_tegra_sms)
                {
                    push (@include_tegra_sms, $def_sm);
                }

                $include_all_desktop = 1;
                $include_all_tegra = 1;
            }
        }
    }

    my $from_sms;
    if ($sm_data->{from})
    {
        $from_sms = $sm_data->{from}[0];
        $from_sms =~ s/^\s+|\s+$//g;
        $from_sms =~ s/\.//g; 
        my $check_from = 0;

        for my $def_sm (@supported_desktop_sms)
        {
            if ($from_sms eq $def_sm)
            {
                $check_from = 1;
            }
            push (@include_desktop_sms, $def_sm) if ($check_from);
        }

        $check_from = 0;
        for my $def_sm (@supported_tegra_sms)
        {
            if ($from_sms eq $def_sm)
            {
                $check_from = 1;
            }
            push (@include_tegra_sms, $def_sm) if ($check_from);
        }

        $include_all_desktop = 1;
        $include_all_tegra = 1;
    }

    @include_desktop_sms = UniqArray(@include_desktop_sms);
    @include_tegra_sms = UniqArray(@include_tegra_sms);
    @exclude_desktop_sms = UniqArray(@exclude_desktop_sms);
    @exclude_tegra_sms = UniqArray(@exclude_tegra_sms);

    die "Must have only all SMs included, or only all SMs excluded." unless ($include_all_desktop ^ $exclude_all_desktop);
    die "Must have only all SMs included, or only all SMs excluded." unless ($include_all_tegra ^ $exclude_all_tegra);

    for my $sm (@include_desktop_sms)
    {
        push (@desktop_sms, $sm);
    }

    if ($include_all_desktop)
    {
        for (@exclude_desktop_sms)
        {
            my $count = scalar @desktop_sms;
            my $index = 0;
            $index++ until @desktop_sms[$index] eq $_ or $index >= $count;
            splice(@desktop_sms, $index, 1);
        }
    }

    for my $sm (@include_tegra_sms)
    {
        push (@tegra_sms, $sm);
    }

    if ($include_all_tegra)
    {
        for (@exclude_tegra_sms)
        {
            my $count = scalar @tegra_sms;
            my $index = 0;
            $index++ until @tegra_sms[$index] eq $_ or $index >= $count;
            splice(@tegra_sms, $index, 1);
        }
    }

    if ($sm_data->{to})
    {
        my $toSM = $sm_data->{to}[0];
        $toSM =~ s/^\s+|\s+$//g;
        $toSM =~ s/\.//g; 

        my $index;
        for my $i (0 .. $#desktop_sms)
        {
            if ($desktop_sms[$i] eq $toSM)
            {
                $index = $i;
            }
        }

        splice (@desktop_sms, $index + 1, ((scalar @desktop_sms) - $index), );
        for my $i (0 .. $#tegra_sms)
        {
            if ($tegra_sms[$i] eq $toSM)
            {
                $index = $i;
            }
        }
        splice (@tegra_sms, $index + 1, ((scalar @tegra_sms) - $index), );
    }

    return (\@desktop_sms, \@tegra_sms);
}

sub UniqArray
{
    my %seen;
    my @return_array;

    foreach (@_)
    {
        unless ($seen{$_})
        {
            push @return_array, $_;
            $seen{@_} = 1;
        }
    }

    return @return_array;
}
