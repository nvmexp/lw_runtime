#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#


#***************************************************************************#
#                        Module : suite_tree                                #
#       Class definition for the node that abstracts the Unit Test          #
#              Tree structure for the arrangement of suites                 #
#                                                                           #
#***************************************************************************#

#
# class representing an element in a tree 
# an object of UnitSuite would be present 
# in the node of the element of the tree
#
class SuiteTree

	# suite present in this node node 
	attr_reader :node

	#parent of this node
	attr_reader :parent

	#children of this node
	attr_reader :children

	# should the tests in this suite be compiled?
	attr_reader :bCompile

	# constructor
	def initialize(name, isEnabled = FALSE,returnType = nil, isStatic= nil, headerPath = nil)

		@node = UnitSuite.new(name, isEnabled, returnType, isStatic, headerPath)
		@parent = nil
		@children = []
		@bCompile = true

	end

	def setParent(parent)

		@parent = parent

	end

	#
	# add this child if of type SuiteTree to the children of 
	# this object, else if it's a string then a new object is created 
	# and added as child
	#
	def add(child, isEnabled = FALSE)

		if child.class != SuiteTree
		child = SuiteTree.new(child, isEnabled)
		end

		# add this child into the children array
		@children << (child)

		child.setParent(self)

	end


	# traverses the tree in depth first fashion
	def each(&block)

		if block_given?

			yield self

		end

	@children.each {|child| child.each(&block) }

	end

	# prunes the tree at the highest level for not enabled node
	def forEachEnabled(&block) 

		if block_given?
		
			yield self if self.node.isEnabled
			
		end

		@children.each {|child| child.forEachEnabled(&block) if child.node.isEnabled }

	end

	# finds the node with the given name
	def find(name)

		#
		# Find the node with the given name
		# also, check if there exists a suite with (dir/file/suite)_name
		#
		self.each do |suite|

			if suite.node.suiteName =~ /#{name}/i  or \
			   suite.node.suiteName =~ /dir_#{name}/i or \
			   suite.node.suiteName =~ /file_#{name}/i or \
			   suite.node.suiteName =~ /suite_#{name}/i

				return suite

			end

		end

		# still name was not found , check if this is of type parentName_"name"
		self.each do |suite|

			if suite.parent

				if suite.node.suiteName =~ /#{suite.parent.node.suiteName}_#{name}/

					return suite

				end

			end

		end

		return nil

	end

	# disable compilation of the tests in this suite
	def disableCompile

		@bCompile = false

	end

	#
	# check if this suite should be skipped while compiling
	# if a direct ancestor is not to be compiled then this
	# suite would not be compiled whatever may be its bCompile status
	#
	def shouldBeCompiled?

		if @parent == nil

			return true

		elsif @parent.bCompile == false

			return false

		else

			return @parent.shouldBeCompiled?

		end

	end

end # SuiteTree