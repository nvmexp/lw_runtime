myParent = $resman.find("dir_fifo_fermi") 

	raise "\n Error : Parent Suite can't be Found \n" if myParent == nil

	# the path of the folder where unittests of this suite would be present
	suiteSrcPath = "diag/unittest/resman/suites/kernel/fifo/fermi/fifog100"

	file_fifog100 = (SuiteTree.new("file_fifog100", TRUE))
	
	# @@fifoServiceTop_GF100@@ BEGINS
	# create a new suite with name fifoServiceTop_GF100
	fifoServiceTop_GF100 = (SuiteTree.new("fifoServiceTop_GF100", TRUE, "RM_STATUS", nil))
	
	# add the new suite fifoServiceTop_GF100 to its parent suite
	file_fifog100.add(fifoServiceTop_GF100)
	# @@fifoServiceTop_GF100@@ ENDS

	# @@Common_Def_Section@@
	# All suite additions must happen before the above comment i.e prior to Common_Def_Section

	# add the file level suite to Dir level suite
	myParent.add(file_fifog100)

	$thisSuite = file_fifog100

	#
	# update the unit test source location for all tests
	# if any test doesn't reside in this expected Location
	# then override after this loop
	#
	$thisSuite.each do |test|

			test.node.testSrcPath = suiteSrcPath + "/ut_" + test.node.suiteName + ".c"

	end

	#
	# refer https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Architecture/RM_Test_Infrastructure/RM_Unit_Test_Framework/Dolwmentation_on_RM_unit_Test_Infra
	# for more information on defintion generation/modification
	#
	
