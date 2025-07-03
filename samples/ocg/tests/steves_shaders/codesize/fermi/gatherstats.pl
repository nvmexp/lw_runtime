# Handle Arguments
my(%arg) = doarg(@ARGV);
local($verbose) = $arg{'verbose'};

# Get all the files that need to be processed
my(@infiles) = get_parseasm_filenames($arg{'files'});

# Gather metrics and log to file
gather_parseasm_metrics($arg{'outfile'},@infiles);
logmsg("CSV output stored in [$arg{'outfile'}].");

# This function is to gather all the filenames to be parsed
sub get_parseasm_filenames {
	my($filestr) = shift;
	my(@match) = split(/,/,$filestr);
	my(@infiles);

	# Determine which files and extensions to match on
	my($lit) = "::"; # holds matches of literal filenames
	my(@ext); # holds extenstion reg expresions
	foreach my $m (@match) {
		my($first,@last); ($first,@last) = split(/\./,$m);
		my($last) = pop(@last);
		if( $last eq "*") { die "ERROR: $m not allowed!\n"; }
		if( $m eq "*.*") { die "ERROR: \*.\* not allowed!\n"; }
		if( $m =~ /\*/ ) { # only supporting *.sss notation
			$m =~ s/\*//g;
			$m .= "\$";
			$m = "\\".$m;
			push(@ext,$m);
		}
		else {
			$lit .= $m."::";
		} #endif patter
	} #endfor match

	# Get list of files
	opendir(DIR,'.') or die "ERROR: Can't open directory!\n";
	my(@files) = readdir(DIR);
	closedir(DIR);

	# Match files
	foreach my $fn (@files) {
		#if( $fn =~ /\.csv$/) {
		my($tmp) = "::".$fn."::";
		if( $lit =~ $tmp) {push(@infiles,$fn);}
		else {
			foreach my $e (@ext) {
				if( $fn =~ /$e/ ) {push(@infiles,$fn);}
			} #endfor e
		} #endif


	} #endfor fn
	return(@infiles);
}

# gather_parseasm_metrics($<outfile>,@<infiles>);
# $<outfile> - This is the file where we will write the csv data to
# @<infiles> - Array of filenames to parse.
sub gather_parseasm_metrics {
	my($outfile) = shift;
	my(@files) = @_;
	my($c) = 0; # counter

	my(%keyindex,@keyorder); # keep track of metrics to record
	my(@parsedfile_hashes); # Stores references to hash for each parsed file

	# csv hates commas in values
	my($comma_replacement) = ";"; # replace commas in values with this

	foreach my $filename (@files) {
		logmsg("Processing [$filename].");
		my(%met) = parse_parseasm_file($filename,$comma_replacement,\@keyorder,\%keyindex);
		$c++;
		$met{'filename'} = $filename;
		push(@parsedfile_hashes,\%met);
		#print ">METRICS\n"; showass(%met); print ">END\n";
	} #endfor filename

	my(@metrickeys); (@metrickeys) = sort(@keyorder);
	unshift(@metrickeys,"filename");

	open(WRITE,">$outfile") or die "ERROR: Can't open [$outfile] for writing!\n";
	my($hdr) = join(",",@metrickeys);
	print WRITE "$hdr\n";

	foreach my $ref (@parsedfile_hashes) {
		#print "ref[$ref]; fn:[$$ref{'filename'}]\n";
		my($outline) = "";
		foreach my $key (@metrickeys) {
			#my($val) = $$ref{$key};
			#print "$key: $val\n";
			$outline .= $$ref{$key}.",";
		} #endfor key
		chop($outline);
		print WRITE $outline."\n";
	} #endfor ref
	close(WRITE);

	my($numkeys); $numkeys = @metrickeys;
	logmsg("$numkeys unique c-style identifier(s) found.");
	logmsg("$c files processed.");
} # gather_parseasm_metrics

# (%<metrics_hash>) = parse_parseasm_file($<filename>,$<commareplace>,<keyorder>,<keyindex>);
# IN
# $<filename> - File to be parsed
# $<commareplace> - Replace commas in the values with this string
# <keyorder> - Reference to an array that tracks keymetric names as they are found
# <keyindex> - Reference to a hash that keeps track of which keys have already been found
# OUT
# %<metrics_hash> - key/value pairs of all the idents that are found in the file
sub parse_parseasm_file {
	my($fn) = shift;
	my($comma_replacement) = shift;
	my($keyorder) = shift;
	my($keyindex) = shift;
	my($c) = 0; # counter
	my($lc) = 1;
	my(%met,$line);
	open(READ,$fn) or die "ERROR: Can't open [$fn] for reading!\n";
	$line = <READ>;
	while ($line ne "") {
		chop($line);
		if(substr($line,0,3) eq "# [") {
			$line =~ s/^\# //g; # get rid of the start of line
			$line = escape_quoted_commas($line,$comma_replacement); # get ready to split on ,
#			my(@parts) = split(/,/,$line); # deal with multiple hits on 1 line
			my(@parts) = splitidents($line); # deal with multiple hits on 1 line
			foreach my $p (@parts) {
				# Cleanup metric element
				$p =~ s/$comma_replacement/\,/g; # replace commas
				$p =~ s/^ +//; # remove preceeding spaces
				$p =~ s/^\[//; # remove preceeding bracket
				$p =~ s/$\]//g; # remove trailing bracket

				# Prepare Key and Value for Hash entry
				my($key,$val);
				($key,$val) = split(/=/,$p);
				$key =~ s/ //g; # No spaces in key
				$val =~ s/^ +//; # remove preceeding spaces
				# Keeping the quotes keeps excel from screwing up the csv values
				#$val =~ s/^\"//; # remove preceeding quote
				#$val =~ s/$\"//g; # remove trailing quote

				# Generate Hash Entry
				$met{$key} = $val;
				$c++;
				
				# Keep track of metrics
				if( !$$keyindex{$key} ) {
					$$keyindex{$key}++;
					push(@$keyorder,$key);
				} #endif


				#print "$p\n";
			} #endfor p
		} #endif

		$line = <READ>;
		$lc++;
	} #end while
	close(READ);
	logmsg("     $c c-style identifier(s) found in [$fn] - $lc lines long.");
	return(%met); # return hash
} # parse_parseasm_file


sub escape_quoted_commas {
        my($line) = shift;
        my($escapewith) = shift;
        my($escapeon) = shift;
	if( $escapeon eq "") { $escapeon = "\""};
        my($escape) = ",";
        my($ret);
        my($qtflag) = 0;
        my($i); for($i=0;$i<length($line);$i++) {
                my($ch) = substr($line,$i,1);
                if( $ch eq $escapeon && !$qtflag ) {$qtflag = 1;}
                elsif( $ch eq $escapeon && $qtflag ) {$qtflag = 0;}
                if( $qtflag && $ch eq $escape) { $ret = $ret.$escapewith; }
                else {$ret=$ret.$ch;}
        } # end fori
        return($ret);
} # escape_quoted_commas

sub logmsg {
	my($msg) = shift;
	if( $verbose ) { print "$msg\n"; }
} # logmsg

sub usage {
	my($u) = "$0: Extracts data from parseasm files.\n\n";
	$u .= "usage: $0 <filename> | <*.ext> [<filename>... <*.ext>...]\n";
	$u .= "\t<filename>...\tfilename(s) to parse c-like identifiers from.\n";
	$u .= "\t<*.ext>...\textensions to match to parse c-like identifiers from.\n\n";
	$u .= "\tCSV output is written to parseasm.csv\n";
	logmsg($u);
}

sub splitidents {
	my($str) = shift;
	my($out);
	my($fl) = 0;
	my($i); for($i=0;$i<length($str);$i++) {
		my($ch) = substr($str,$i,1);

		if( $ch eq "\[") {
			$fl = 1;
		} #endif

		if( $fl ) { $out .= $ch; }

		if( $ch eq "\]") {
			$fl = 0;
		} #endif



	} #end for
	$out =~ s/\]/\,/g;
	$out =~ s/\[//g;
	$out =~ s/\,$//;
	my(@parts) = split(/,/,$out);
	return(@parts);
}

# -o <outputfile> -f <filename,filename,ext,ext,filename,ext,... -v
# <outputfile> - Default is parseasm.csv
# <filenamelist> - Default is *.s
# -v by default is off
# outfile,verbose,files
sub doarg {
	my(@arg) = @_;
	my(%ret);

	if( $arg[0] eq "") { die usage();}

	my($a); for($a=0;$a<@arg;$a++) {

		if(    substr($arg[$a],0,2) eq "-o" ) { $ret{'outfile'} = $arg[$a+1]; }
		elsif( substr($arg[$a],0,2) eq "-v" ) { $ret{'verbose'} = 1; }
		elsif( substr($arg[$a],0,2) eq "-f" ) { $ret{'files'} = $arg[$a+1]; }
		elsif( substr($arg[$a],0,2) eq "-h" || $arg[$a] eq "\/h") { $ret{'help'} = 1; }

	} #endfor a

	if( $ret{'help'} ) {
		my($usage);
		$usage = "\n$0 -verbose -help -o <outfile> -f <filelist>\n";
		$usage .= "\n\tExtracts parseasm c-style identifiers into a comma seperated file.\n";
		$usage .= "\n\t-h\t\tDisplays Usage.\n";
		$usage .= "\t-verbose\tEnables messages to STDOUT.\n";
		$usage .= "\t-o <outfile>\tSets the outfile. parseasm.csv is default.\n";
		$usage .= "\t-f <filelist>\tList of files seperated by commas to parse c-style identifiers from.\n";
		$usage .= "\t\t\tCan specify literal filenames or file extensions.\n";
		$usage .= "\t\t\t*.* is not allowed.\n";
		$usage .= "\t\t\tfile.* is not allowed.\n";
		$usage .= "\t\t\tOnly finds files in the current direct.\n";
		$usage .= "\n\tExample:\n\n\t$0 -o parseasmdata.csv -f *.s\n";
		die $usage;
	} #endif help

	# Set defaults
	if( $ret{'outfile'} eq "") { $ret{'outfile'} = "parseasm.csv"; }
	if( $ret{'verbose'} eq "") { $ret{'verbose'} = 0; }
	if( $ret{'files'}   eq "") { $ret{'files'} = "*.s"; }

	# Check that outfile is writable before we work too hard
	open(WRITE,">".$ret{'outfile'}) or die "ERROR: Cannot open [$ret{'outfile'}] to write!\n"; 
	close(WRITE);

	return(%ret);
} # doarg

