#!/usr/bin/perl

use FindBin qw($Bin);

use lib "$Bin/lib";
use lib "$Bin/lib";

use strict;
use Carp qw(croak);
use File::Path qw(mkpath);
use File::Basename qw(basename dirname);
use Data::Dumper qw(Dumper);
use List::Util qw(sum);
use LWPU::Log qw(LogV);
use LWPU::Log::StderrLogger;
use LWPU::Config;
use LWPU::ProgressBar;
use LWPU::Hadoop::WebHDFS;
use LWPU::Hadoop::API::JobConf qw(:methods
                                    :job_type_enum
                                    :job_status_enum
                                    :job_priority_enum
                                    :task_type_enum
                                    :task_status_enum);

LWPU::Log::AddLogger (new LWPU::Log::StderrLogger);
LWPU::Log::SetLwrrentModule ('HADOOP');

LWPU::Config::ExportToElw();
our $WebHDFS = new LWPU::Hadoop::WebHDFS();


{
  my $Date   = $ARGV[0] || die 'you should specify date';
  my $Script = 'RawAnselToTSV.py';
  my $Window  = $ARGV[1] || die 'Window not specified';

  my $Timestamp = LWPU::Time->new($Date."T00:00:00Z")->Epoch();
  my @Path = map { LWPU::Time->new($Timestamp-(86400 * $_))->FormatGM('%Y-%m-%d') } (0..($Window-1));

  my $PATH = eval("[\"" . join('","', map { '/gfe/ansel/raw/json/prod/'.$_ } @Path) . "\"]");

  my $Name = "Raw Ansel Events (TSV): ".$Date." x ".$Window;
  
  my $JobConf =
    {
      Name       => $Name,
      Type       => &JOB_TYPE_STREAMING,
      Priority   => &JOB_PRIORITY_HIGH,
      Processor  => $Script,
      InputPaths => $PATH,
      OutputPath => "RawAnselToTSV/".$Date."_".time(),
      Properties =>
        {
            'mapreduce.map.memory.mb'                                  => 2048,
            'mapreduce.map.java.opts'                                  => '-Xmx1024m',
            'mapred.input.format.class'                                => 'com.lwpu.hadoop.CombinedTextInputFormat',
            'mapreduce.input.fileinputformat.split.maxsize'            => 268_435_456, # 256M
            'dfs.blocksize'                                            => 67_108_864, # 64M
            'dfs.client.block.write.retries'                           => 15,
            'stream.map.output'                                        => 'rawbytes',
            'mapreduce.map.output.compress.codec'                      => 'org.apache.hadoop.io.compress.SnappyCodec',
            'mapreduce.job.reduces'                                    => 1,
            'mapreduce.reduce.memory.mb'                               => 2048,
            'mapreduce.reduce.java.opts'                               => '-Xmx1024m',
            'mapreduce.job.reduce.slowstart.completedmaps'             => 0.5,
            'mapreduce.reduce.shuffle.parallelcopies'                  => 3,    # 5 The default number of parallel transfers run by reduce during the
                                                                                # copy(shuffle) phase.
            'mapreduce.reduce.shuffle.input.buffer.percent'            => 0.50, # 0.70 The percentage of memory to be allocated from the maximum heap size to
                                                                                # storing map outputs during the shuffle.
            'mapreduce.reduce.shuffle.merge.percent'                   => 0.50, # 0.66 The usage threshold at which an in-memory merge will be initiated,
                                                                                # expressed as a percentage of the total memory allocated to storing in-memory
                                                                                # map outputs, as defined by mapreduce.reduce.shuffle.input.buffer.percent.
            'mapreduce.output.fileoutputformat.compress'               => 'true',
            'mapreduce.output.fileoutputformat.compress.codec'         => 'org.apache.hadoop.io.compress.GzipCodec',        
},
    };

  UploadScript($JobConf);
  my $JobId = LaunchJob($JobConf);
  LogV 1, "Submitted Job '$JobId'";

  my $Summary;

  LogV 5, "Getting list of tasks";
  my(%Tasks);
  do
  {
    sleep(5);
    %Tasks = 
      map
      {
        $_->{Id} =>
          {
            Type     => $_->{Type},
            Progress => $_->{Progress}
          }
      }
      @{ SearchTasksShort( << "END" ) };
<Search>
 <Expr>
  <And>
   <Equals>
    <Field name='JobId'/>
    <Value>$JobId</Value>
   </Equals>
   <In>
    <Field name='Type'/>
    <Value>@{[&TASK_TYPE_MAP]}</Value>
    <Value>@{[&TASK_TYPE_REDUCE]}</Value>
   </In>
  </And>
 </Expr>
 <Sort>
  <Field name='Id' order='DESC'/>
 </Sort>
</Search>
END
  }
  while(!scalar keys %Tasks);

  LogV 1, "Found ".scalar(keys(%Tasks))." tasks for job '$JobId'";

  my $ProgressBar = LWPU::ProgressBar->new(
    title       => $JobConf->{Name},
    count       => 100,
    style       => 'BAR',
    char_filled => '#',
    char_space  => '=');

  do
  {
    sleep(5);

    my @TaskIds =
      grep { $Tasks{ $_ }->{Progress} < 1 }
      keys %Tasks;

    if(@TaskIds)
    {
      map
      {
        $Tasks{ $_->{Id} }->{Progress} = $_->{Progress};
      } @{ GetTasksShort(\@TaskIds) };
    }

    my @Progress =
      map { $_->{Progress} }
      values %Tasks;

    $ProgressBar->Update(int(sum(@Progress)/scalar(@Progress)*100))
      if @Progress > 0;

    $JobConf = GetJobShort($JobId);

    warn "Job '$JobId' has been failed"
      if $JobConf->{Status} == &JOB_STATUS_FAILED;
    warn "Job '$JobId' has been killed"
      if $JobConf->{Status} == &JOB_STATUS_KILLED;
    LogV 1, "Job '$JobId' has been completed"
      if $JobConf->{Status} == &JOB_STATUS_SUCCESS;
  }
  while($JobConf->{Status} <= &JOB_STATUS_RUNNING);
  $ProgressBar->Done();

  my $JobConf = GetJob($JobId);

  # Cleanup
  $WebHDFS->rm($JobConf->{Processor});
  LogV 5, "Removed script '$JobConf->{Processor}' from hdfs";

  DownloadOutputs($JobConf)
    if $JobConf->{Status} == &JOB_STATUS_SUCCESS;
}



sub UploadScript
{
  my ($Job) = @_;

  LogV 5, "Removing files from hdfs";
  $WebHDFS->rm($Job->{OutputPath}, relwrsive => 1);
  
  LogV 5, "Loading script '$Job->{Processor}'";
  my $Data = '';
  open(*FH, $Job->{Processor})
    or croak("Cannot open script: $Job->{Processor}: $!");
  {
    local $/ = undef;
    $Data = <FH>;
  }
  close(*FH);

  LogV 5, "User: '$WebHDFS->{Username}'";
  
  my $Processor = $WebHDFS->home . '/' . basename($Job->{Processor});
  LogV 5, "Copying script into hdfs '$Processor'";
  $WebHDFS->put($Processor, \$Data,
                replication => 3,
                permission => '777',
                overwrite => 1);
  $Job->{Processor} = basename($Processor);
  return;
}

sub LaunchJob
{
  my ($Job) = @_;

  LogV 5, "Preparing Job config";
  $Job->{Properties} =
    [
      (($Job->{Properties}->{'mapred.input.format.class'} eq 'com.lwpu.hadoop.WholeFileInputFormat') ?
        ({
            Name => 'mapred.map.child.ulimit',
            Value => -1,
          },
          {
            Name => 'mapred.reduce.child.ulimit',
            Value => -1,
          }) : ()),
      map
      {
        {
          Name => $_,
          Value => $Job->{Properties}->{$_}
        }
      } keys %{ $Job->{Properties} },
    ];
  LogV 5, "Submitting new job";
  return AddJob($Job);
}

sub DownloadOutputs
{
  my ($Job) = @_;

  LogV 5, "Getting list of output files";
  my @OutputFiles =
    grep { $_->{Name} ne '_SUCCESS' }
    @{ $WebHDFS->ls($Job->{OutputPath}) };
  if(@OutputFiles == 0)
  {
    LogV 1, "No output files found";
    return;
  }

  my $TotalSize = sum(map { $_->{Size} } @OutputFiles);
  LogV 2, sprintf('There are %d file(s) of %.2fMB in output directory',scalar(@OutputFiles),($TotalSize/1024/1024));

  if($TotalSize > (500 * 1024 * 1024))
  {
    warn "Skipping download, because outputs size greater than 500MB";
    return;
  }
  mkpath(basename($Job->{OutputPath}));
  my $ProgressBar = LWPU::ProgressBar->new(
    title       => $Job->{OutputPath},
    count       => scalar @OutputFiles,
    style       => 'BAR',
    char_filled => '#',
    char_space  => '=');
  for(0..($#OutputFiles))
  {
    next if $OutputFiles[ $_ ]->{Name} eq '_SUCCESS';
    my $Data = $WebHDFS->get($Job->{OutputPath} . '/' . $OutputFiles[ $_ ]->{Name});
    open(*FH,'>'.basename($Job->{OutputPath}).'/'.$OutputFiles[ $_ ]->{Name}) or croak("Cannot open '".basename($Job->{OutputPath}).'/'.$OutputFiles[ $_ ]->{Name}."': $!");
    binmode(FH,':raw');
    print FH $$Data;
    close(*FH);
    $ProgressBar->Update($_);
  }
  $ProgressBar->Done();
  #LogV 5, "Removing files from hdfs";
  #$WebHDFS->rm($Job->{OutputPath}, relwrsive => 1);
  LogV 1, "Output files succesfully downloaded to '".basename($Job->{OutputPath})."/'";
  return;
}
