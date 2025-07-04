/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/*
 * This file is almost a complete re-write for Open MPI compared to the
 * original mpiJava package. Its license and copyright are listed below.
 * See <path to ompi/mpi/java/README> for more information.
 */
/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */
/*
 * File         : User_function.java
 * Author       : Xinying Li
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.4 $
 * Updated      : $Date: 1999/09/13 16:14:30 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */

package mpi;

import java.nio.*;

/**
 * Java equivalent of the {@code MPI_USER_FUNCTION}.
 */
public abstract class UserFunction
{
	/**
	 * User-defined function for a new {@code Op}.
	 * @param ilwec    		array of values to combine with {@code inoutvec} elements
	 * @param inOutVec 		in-out array of aclwmulator locations
	 * @param count    		number of items in arrays
	 * @param datatype 		type of each item
	 * @throws MPIException Signals that an MPI exception of some sort has oclwrred.
	 */
	public void call(Object ilwec, Object inOutVec, int count, Datatype datatype)
			throws MPIException
	{
		throw new UnsupportedOperationException("Not supported yet.");
	}

	/**
	 * User-defined function for a new {@code Op}.
	 * @param in       		direct byte buffer to combine with {@code inOut} buffer
	 * @param inOut   		in-out direct byte buffer of aclwmulator locations
	 * @param count    		number of items in buffers
	 * @param datatype 		type of each item
	 * @throws MPIException Signals that an MPI exception of some sort has oclwrred.
	 */
	public void call(ByteBuffer in, ByteBuffer inOut, int count, Datatype datatype)
			throws MPIException
	{
		switch(datatype.baseType)
		{
		case Datatype.BYTE:
			vCall(in, inOut, count, datatype);
			break;
		case Datatype.CHAR:
			vCall(in.asCharBuffer(), inOut.asCharBuffer(), count, datatype);
			break;
		case Datatype.SHORT:
			vCall(in.asShortBuffer(), inOut.asShortBuffer(), count, datatype);
			break;
		case Datatype.INT:
			vCall(in.asIntBuffer(), inOut.asIntBuffer(), count, datatype);
			break;
		case Datatype.LONG:
			vCall(in.asLongBuffer(), inOut.asLongBuffer(), count, datatype);
			break;
		case Datatype.FLOAT:
			vCall(in.asFloatBuffer(), inOut.asFloatBuffer(), count, datatype);
			break;
		case Datatype.DOUBLE:
			vCall(in.asDoubleBuffer(), inOut.asDoubleBuffer(), count, datatype);
			break;
		case Datatype.PACKED:
			vCall(in, inOut, count, datatype);
			break;
		default:
			throw new IllegalArgumentException("Unsupported datatype.");
		}
	}

	private void vCall(ByteBuffer in, ByteBuffer inOut,
			int count, Datatype datatype) throws MPIException
	{
		int    extent   = datatype.getExtent();
		byte[] ilwec    = new byte[count * extent],
				inOutVec = new byte[count * extent];

		in.get(ilwec);
		inOut.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOut.clear();
		inOut.put(inOutVec);
	}

	private void vCall(CharBuffer inBuf, CharBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int    extent   = datatype.getExtent();
		char[] ilwec    = new char[count * extent],
				inOutVec = new char[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

	private void vCall(ShortBuffer inBuf, ShortBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int     extent   = datatype.getExtent();
		short[] ilwec    = new short[count * extent],
				inOutVec = new short[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

	private void vCall(IntBuffer inBuf, IntBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int   extent   = datatype.getExtent();
		int[] ilwec    = new int[count * extent],
				inOutVec = new int[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

	private void vCall(LongBuffer inBuf, LongBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int    extent   = datatype.getExtent();
		long[] ilwec    = new long[count * extent],
				inOutVec = new long[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

	private void vCall(FloatBuffer inBuf, FloatBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int     extent   = datatype.getExtent();
		float[] ilwec    = new float[count * extent],
				inOutVec = new float[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

	private void vCall(DoubleBuffer inBuf, DoubleBuffer inOutBuf,
			int count, Datatype datatype) throws MPIException
	{
		int      extent   = datatype.getExtent();
		double[] ilwec    = new double[count * extent],
				inOutVec = new double[count * extent];

		inBuf.get(ilwec);
		inOutBuf.get(inOutVec);
		call(ilwec, inOutVec, count, datatype);
		inOutBuf.clear();
		inOutBuf.put(inOutVec);
	}

} // UserFunction
