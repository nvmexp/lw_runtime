use FindBin qw($Bin);
use Data::Dumper;
# expand pathname to long pathname
$Bin = Win32::GetLongPathName($Bin);

use lib "$Bin/lib";
use lib "$Bin/../lib";
use lib "$Bin/../../lib";
use lib "$Bin/../../common/lib";

#use Results;
use File::Copy;

my $LF = "$Bin/output.txt";
&logToFile($LF, "BEGIN");
#chdir("..\\..\\");
unlink("result.txt") if -e "result.txt";
#Command line arguments to the application

my $fname = "Vsync_Off";
$cmd = "StutterLatencyTestsApp.exe $fname 0";
logOutput("CMD:[$cmd]");
my $cmd_out= `$cmd`;

my $fname = "Vsync_On";
$cmd = "StutterLatencyTestsApp.exe $fname 1";
logOutput("CMD:[$cmd]");
my $cmd_out= `$cmd`;

sub logOutput()
{
   local($message) = $_[0];
   &logToFile($LF, $message);
}

sub logToFile()
{
    local($logfile, $message) = ($_[0], $_[1]);
    open( LOGFILE , ">>$logfile");
    print LOGFILE  &getTimestamp() . "$message\n";
    close(LOGFILE);
}
sub getTimestamp
{
   my ($sec, $min, $hr, $day, $mon, $year) = localtime;
   sprintf("[%02d/%02d/%04d %02d:%02d:%02d]", $day, $mon + 1, 1900 + $year, $hr, $min, $sec);
}
