#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define MAX(A,B)        (((A)>(B))?(A):(B))

void qsort_(double *val,int *ind, int *stack, int *n);


void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
            double *w, double *work, int *lwork, int *info, int l1, int l2);
void dgemm_(char *transa, char *transb,int *m, int *n,int *k, double *alpha,
	    double *A, int *lda, double *B, int *ldb, double *beta, double *C,
	    int *ldc, int l1, int l2);
void dcopy_(int *n, double *dx, int *incx, double *dy, int *incy);
double dnrm2_(int *n, double *dx, int *incx);




int jacobi(double **A, int n, double **V, int block_size, double tol, double *hist, int *iter)
{

   /*  
       Jacobi-like method for computing eigenvalues and eigenvectors of a
       symmetric matrix

       Input
       -----
       A           symmetric dense matrix of size n x n
       n           matrix size
       block_size  block size, Jacobi method solves cyclically sub problems of
                   size 2*block_size x 2*block_size
       tol         termination threshold. Jacobi method stops whenever
                   sum_{i~=j} A(i,j)^2 <= tol * sum_{i} A(i,i)^2
       iter        on input, maximum number of iteration steps

       Output
       ------
       return value 0 success, 1 failure. If Jacobi converged within "iter"
                    steps, the the return value will be zero, otherwise it
		    will be 1
       A            diagonal matrix with real eigenvalues
       V            matrix of (orthonormal) eigenvectors
       hist         convergence history of the Jacobi sweeps
                    (sum_{i~=j} A(i,j)^2) / (sum_{i} A(i,i)^2)
       iter         on output, number of Jacobi sweeps
   */
   int i,j, ii,jj,     // counters
       k=2*block_size, // size of the local systems
       m,
       r,s,            // indices for blocks to be considered
       mythreadnum,    // my thread ID
       max_iter=*iter, // maximum number of sweeps
       lwork=MAX(k*k-k,3*k-1), info; // variables for dsyev
   
   double *pA,*pV,             // pointers for A and V
          alpha=1.0, beta=0.0, // scalars for dgemm
          **dbuff, *pdbuff;    // buffer and pointer for dgemm

   
   // store size of the problem in case we have to enlarge it  
   int nn=n;

   // check whether the block size fits into the matrix
   int l=nn % (2*block_size);
   // block size does not exactly fit, add zero rows and columns
   if (l) {
      m=2*block_size-l;
      n+=m;
      *A=(double *)realloc(*A,(size_t)n*n*sizeof(double));
      *V=(double *)realloc(*V,(size_t)n*n*sizeof(double));

      // shift entries of A such that in every column we leave m spaces
      // pointer to the beginning of the last column of A
      ii=m*(nn-1);
      pA=*A+(nn-1)*nn;
      for (j=nn-1; j>=1; j--,ii-=m,pA-=nn)
	  memmove(pA+ii,pA,nn*sizeof(double));
      // clear extended part
      for (j=0; j<nn; j++) {
	  pA=*A+j*n+nn;
	  for (i=nn; i<n; i++)
	      *pA++=0.0;
      } // end for j
      for (j=nn; j<n; j++) {
	  pA=*A+j*n;
	  for (i=0; i<n; i++)
	      *pA++=0.0;
      } // end for j
      // now n is a mutliple of 2*block_size
      printf("augmented n=%d\n",n);
   } // end if

   m=omp_get_max_threads();
   // auxiliary buffer for GEMM, required for every thread
   dbuff=(double **)malloc((size_t)m*sizeof(double *));
   // index buffers for qsort
   int   **ind=(int **)   malloc((size_t)m*sizeof(int *)), *pind;
   int **stack=(int **)   malloc((size_t)m*sizeof(int *)), *pstack;
   for (i=0; i<m; i++) {
       // each buffer must have size n x k
       dbuff[i]=(double *)malloc((size_t)n*k*sizeof(double));
       // each integer array must have size k
       ind[i]  =(int *)malloc((size_t)k*sizeof(int));
       stack[i]=(int *)malloc((size_t)k*sizeof(int));
   } // end for i
   
   // number of parallel problems to be solved simultaneously
   // now n must be a multiple of 2*block_size
   m=n/(2*block_size);
   // initial partitioning (0,1), (2,3), (4,5), ... (2m-2,2m-1)
   int *coord1=(int *)malloc((size_t)m*sizeof(int));
   int *coord2=(int *)malloc((size_t)m*sizeof(int));
   // initial parallel ordering pairs
   for (i=0; i<m; i++) {
       coord1[i]=2*i;
       coord2[i]=2*i+1;
   } // end for i

   // initial eigenvector matrix = I
   pV=*V;
   for (j=0; j<n; j++) {
       for (i=0; i<j; i++)
	   *pV++=0.0;
       *pV++=1.0;
       for (i=j+1; i<n; i++)
	   *pV++=0.0;
   } // end for j
  
   // square sum of the diagonal entries
   double nrm_dgl=0.0, nrm_dgl_loc, mynrm_dgl;
   pA=*A;
   for (j=0; j<n; j++,pA+=n+1) 
       nrm_dgl+=*pA**pA;
   
   // square sum of the off-diagonal entries
   double nrm_offdgl=0.0, nrm_offdgl_loc, mynrm_offdgl;
   pA=*A;
   for (j=0; j<n; j++) {
       for (i=0; i<j; i++,pA++)
           nrm_offdgl+=*pA**pA;
       pA++;
       for (i=j+1; i<n; i++,pA++)
           nrm_offdgl+=*pA**pA;
   } // end for j


   // auxiliary space for m local eigenpair computations
   // local eigenvectors after dsyev, later eigenvalues
   double **DD=(double **)malloc((size_t)m*sizeof(double *)), *pD;
   // copy of the local diagonal entries, stored because their order
   // imposes the order of the eigenvalues
   double **AA=(double **)malloc((size_t)m*sizeof(double *));
   // local eigenvectors after dsyev and reordering
   double **VV=(double **)malloc((size_t)m*sizeof(double *));
   for (i=0; i<m; i++) {
       DD[i]=(double *)malloc((size_t)k*k*sizeof(double));
       AA[i]=(double *)malloc((size_t)k*  sizeof(double));
       VV[i]=(double *)malloc((size_t)MAX(k*k,4*k-1)*sizeof(double));
   } // end for i
   
   *iter=0;
   while (nrm_offdgl>tol*nrm_dgl && *iter<max_iter) {
         hist[*iter]=nrm_offdgl/nrm_dgl;
         *iter=*iter+1;
	 
	 // compute yet another sweep
	 for (i=0; i<2*m-1; i++) {
	     // this loop could in principle be performed in parallel
	     // but multiplication of D must be synchronized, at first
	     // multiplication from the right with VV is done in common
	     // after that, when all multiplications D(:,[I J])*VV are
	     // finished, then all multiplications VV'*D([I J],:) can
	     // be computed
	     // first stage: solve small eigenvalue problem and update norms
	     mynrm_dgl=0.0;
	     mynrm_offdgl=0.0;
#pragma omp parallel for default(none)\
  shared(m,ind,stack,coord1,coord2,A,block_size,DD,n,k,AA,VV,lwork,stdout)\
            private(mythreadnum,pind,pstack,r,s,pA,pD,jj,ii,nrm_dgl_loc,\
		    nrm_offdgl_loc,info,pV)\
	    reduction(+: mynrm_dgl,mynrm_offdgl)
	     for (j=0; j<m; j++) {
	         mythreadnum=omp_get_thread_num();
		 // local index arrays used for qsort
		 pind  =ind[mythreadnum];
		 pstack=stack[mythreadnum];
	         // extract block number (r,s)
	         r=coord1[j]; s=coord2[j];
		 // solve eigenvalue problem of block (r,s)
		 // first sub block  I=r*block_size,...,(r+1)*block_size-1
		 // second sub block J=s*block_size,...,(s+1)*block_size-1
		 // local symmetric matrix block DD=A([I J],[I J]);
		   
		 // 1. DD(1:bs,1:bs)=A(I,I)
		 pA=*A+r*block_size+r*block_size*n;
		 pD=DD[j];
		 for (jj=0; jj<block_size; jj++,pA+=n,pD+=k)
		     memcpy(pD,pA,block_size*sizeof(double));
		 // 2. DD(bs+1:2bs,1:bs)=A(J,I)
		 pA=*A+s*block_size+r*block_size*n;
		 pD=DD[j]+block_size;
		 for (jj=0; jj<block_size; jj++,pA+=n,pD+=k)
		     memcpy(pD,pA,block_size*sizeof(double));
		 // 3. DD(1:bs,bs+1:2bs)=A(I,J)
		 pA=*A+r*block_size+s*block_size*n;
		 pD=DD[j]+block_size*k;
		 for (jj=0; jj<block_size; jj++,pA+=n,pD+=k)
		     memcpy(pD,pA,block_size*sizeof(double));
		 // 4. DD(bs+1:2bs,bs+1:2bs)=A(J,J)
		 pA=*A+s*block_size*n+s*block_size;
		 pD=DD[j]+block_size+block_size*k;
		 for (jj=0; jj<block_size; jj++,pA+=n,pD+=k)
		     memcpy(pD,pA,block_size*sizeof(double));

		 // local square sum of the diagonal entries
		 pD=DD[j];
		 nrm_dgl_loc=0.0;
		 for (jj=0; jj<k; jj++,pD+=k+1) 
		     nrm_dgl_loc+=*pD**pD;

		 //  local square sum of the off-diagonal entries
		 pD=DD[j];
		 nrm_offdgl_loc=0.0;
		 for (jj=0; jj<k; jj++) {
		     for (ii=0; ii<jj; ii++,pD++)
		         nrm_offdgl_loc+=*pD**pD;
		     pD++;
		     for (ii=jj+1; ii<k; ii++,pD++)
		         nrm_offdgl_loc+=*pD**pD;
		 } // end for jj

	         // locally downdate squared norm of the diagonal entries
		 mynrm_dgl   -=nrm_dgl_loc;
		 // locally downdate squared norm of the off-diagonal entries
		 mynrm_offdgl-=nrm_offdgl_loc;

		 // copy diagonal entries of A([I J],[I J])
		 pA=AA[j];
		 pD=DD[j];
		 for (jj=0; jj<k; jj++,pA++,pD+=k+1)
		     *pA=*pD;
		 
		 // solve local eigenvalue problem
		 // [VV{j},DD{j}]=schur(AA);
		 pV=VV[j];
		 pD=DD[j];
                 dsyev_("V","L",&k, pD,&k, pV, pV+k,&lwork, &info, 1,1);
		 if (info<0) {
		    printf("DSYEV: the %d-th argument had an illegal value\n",
			   -info);
		    exit(1);
		 }
		 else if (info>0) {
		    printf("DSYEV: the algorithm failed to converge; %d off-diagonal elements of an intermediate tridiagonal form did not converge to zero.\n", info);
		    exit(1);
		 } // end if-else if
		 
	         // find out which eigenvalues are closest to the diagonal entries
		 // of AA. This is necessary to make sure that VV tends to I when
		 // AA tends to be diagonal
		 // natural ordering
		 for (jj=0; jj<k; jj++)
		     pind[jj]=jj;
		 // sort diagonal entries in ascending order
		 qsort_(AA[j],pind,pstack,&k);
		 // now pind refers to permuting the original diagonal entries
		 // in ascending order
		 // use pstack for inverse permutation
		 for (jj=0; jj<k; jj++)
		     pstack[pind[jj]]=jj;
		 // reorder eigenpairs and eigenvalues according to inverse
		 // permutation
		 // copy reordered eigenvalues to AA[j]
		 pV=VV[j];
		 pA=AA[j];
		 for (jj=0; jj<k; jj++)
		     pA[jj]=pV[pstack[jj]];
		 // copy reordered eigenvectors to VV[j]
		 pV=VV[j];
		 pD=DD[j];
		 for (jj=0; jj<k; jj++,pV+=k)
		     memcpy(pV,pD+pstack[jj]*k,k*sizeof(double));

		 // copy eigenvalues back to DD[j]
		 memcpy(pD,pA,k*sizeof(double));
		 
		 // locally update squared norm of the diagonal entries
		 nrm_dgl_loc=0.0;
		 for (jj=0; jj<k; jj++,pD++) 
		     nrm_dgl_loc+=*pD**pD;
		 mynrm_dgl+=nrm_dgl_loc;
	     } // end for j
	     // end omp parallel for
	     // globally downdate squared norm of the diagonal entries
	     nrm_dgl   +=mynrm_dgl;
	     // globally downdate squared norm of the off-diagonal entries
	     nrm_offdgl+=mynrm_offdgl;
	   
	     // second stage: multiplication from the right
#pragma omp parallel for default(none)\
                         shared(m,dbuff,coord1,coord2,V,block_size,n,k,alpha,\
				beta,VV,A)\
                         private(mythreadnum,pdbuff,r,s,pV,jj,pA)
	     for (j=0; j<m; j++) {
	         mythreadnum=omp_get_thread_num();
		 pdbuff=dbuff[mythreadnum];
	         // extract block number (r,s)
	         r=coord1[j]; s=coord2[j];
		 // first sub block  I=r*block_size,...,(r+1)*block_size-1
		 // second sub block J=s*block_size,...,(s+1)*block_size-1

		 // update eigenvector matrix
		 // V(:,[I J])=V(:,[I J])*VV{j};
		 // copy V(:,[I J]) to dbuff
		 pV=*V+r*block_size*n;
		 for (jj=0; jj<block_size; jj++)
		     memcpy(pdbuff+jj*n,pV+jj*n,n*sizeof(double));
		 pV=*V+s*block_size*n;
		 for (jj=0; jj<block_size; jj++)
 		     memcpy(pdbuff+(jj+block_size)*n,pV+jj*n,n*sizeof(double));

		 // V(:,I) <- 1 * dbuff * VV{j}(:,1:bs) + 0 * V(:,I)
		 dgemm_("n","n", &n,&block_size, &k, &alpha, pdbuff,&n,
			VV[j],&k,
			&beta, *V+r*block_size*n,&n, 1,1);
		 // V(:,J) <- 1 * dbuff * VV{j}(:,bs+1:2bs) + 0 * V(:,J)
		 dgemm_("n","n", &n,&block_size, &k, &alpha, pdbuff,&n,
			VV[j]+block_size*k,&k,
			&beta, *V+s*block_size*n,&n, 1,1);
		 // update eigenvalue matrix
		 // A(:,[I J])=A(:,[I J])*VV{j};
		 // copy A(:,[I J]) to dbuff
		 pA=*A+r*block_size*n;
		 for (jj=0; jj<block_size; jj++)
		     memcpy(pdbuff+jj*n,pA+jj*n,n*sizeof(double));
		 pA=*A+s*block_size*n;
		 for (jj=0; jj<block_size; jj++)
 		     memcpy(pdbuff+(jj+block_size)*n,pA+jj*n,n*sizeof(double));
		 
		 // A(:,I) <- 1 * dbuff * VV{j}(:,1:bs) + 0 * A(:,I)
		 dgemm_("n","n", &n,&block_size, &k, &alpha, pdbuff,&n,
			VV[j],&k,
			&beta, *A+r*block_size*n,&n, 1,1);
		 // A(:,J) <- 1 * dbuff * VV{j}(:,bs+1:2bs) + 0 * A(:,J)
		 dgemm_("n","n", &n,&block_size, &k, &alpha, pdbuff,&n,
			VV[j]+block_size*k,&k,
			&beta, *A+s*block_size*n,&n, 1,1);
	     } // end for j
	     // end omp parallel for

	     // third stage: multiplication of A from the left
#pragma omp parallel for default(none)\
                         shared(m,dbuff,coord1,coord2,A,block_size,k,n,alpha,\
				VV,beta)\
                         private(mythreadnum,pdbuff,r,s,pA,ii,jj)
	     for (j=0; j<m; j++) {
	         mythreadnum=omp_get_thread_num();
		 pdbuff=dbuff[mythreadnum];
	         // extract block number (r,s)
	         r=coord1[j]; s=coord2[j];
		 // first sub block  I=r*block_size,...,(r+1)*block_size-1
		 // second sub block J=s*block_size,...,(s+1)*block_size-1
		 // update eigenvalue matrix
		 // A([I J],:)=VV{j}'*A([I J],:);
		 // copy A([I J],:) to dbuff'
		 pA=*A+r*block_size;
		 ii=1;
		 for (jj=0; jj<block_size; jj++)
		     dcopy_(&n, pA+jj,&n, pdbuff+jj*n,&ii);
		 pA=*A+s*block_size;
		 for (jj=0; jj<block_size; jj++)
		     dcopy_(&n, pA+jj,&n, pdbuff+(jj+block_size)*n,&ii);

		 // A(I,:) <- 1 * VV{j}(:,1:bs)' * dbuff'  + 0 * A(I,:)
		 dgemm_("T","T", &block_size,&n, &k, &alpha, VV[j],&k,
			pdbuff,&n,
			&beta, *A+r*block_size,&n, 1,1);
		 // A(J,:) <- 1 * VV{j}(:,bs+1:2bs)'* dbuff' + 0 * A(J,:)
		 dgemm_("T","T", &block_size,&n, &k, &alpha, VV[j]+block_size*k,&k,
			pdbuff,&n,
			&beta, *A+s*block_size,&n, 1,1);
	     } // end for j
	     // end omp parallel for
	     
	     // fourth stage: diagonal blocks and new coordinates
#pragma omp parallel for default(none)\
                         shared(m,coord1,coord2,A,block_size,n,DD)\
                         private(r,s,pA,pD,jj,ii)
	     for (j=0; j<m; j++) {
	         // extract block number (r,s)
	         r=coord1[j]; s=coord2[j];
		 // first sub block  I=r*block_size,...,(r+1)*block_size-1

		 // update eigenvalue matrix
		 // A([I J],[I J])=DD{j}
		 // 1. A(I,I)
		 pA=*A+r*block_size+r*block_size*n;
		 pD=DD[j];
		 for (jj=0; jj<block_size; jj++,pA+=n) {
		     for (ii=0; ii<jj; ii++)
		         pA[ii]=0.0;
		     pA[jj]=pD[jj];
		     for (ii=jj+1; ii<block_size; ii++) 
		         pA[ii]=0.0;
		 } // end for jj
		 // 2. A(J,I)
		 pA=*A+s*block_size+r*block_size*n;
		 for (jj=0; jj<block_size; jj++,pA+=n) {
		     for (ii=0; ii<block_size; ii++) 
		         pA[ii]=0.0;
		 } // end for jj
		 // 3. A(I,J)
		 pA=*A+r*block_size+s*block_size*n;
		 for (jj=0; jj<block_size; jj++,pA+=n) {
		     for (ii=0; ii<block_size; ii++) 
		         pA[ii]=0.0;
		 } // end for jj
		 // 4. A(J,J)
		 pA=*A+s*block_size+s*block_size*n;
		 pD=DD[j]+block_size;
		 for (jj=0; jj<block_size; jj++,pA+=n) {
		     for (ii=0; ii<jj; ii++)
		         pA[ii]=0.0;
		     pA[jj]=pD[jj];
		     for (ii=jj+1; ii<block_size; ii++) 
		         pA[ii]=0.0;
		 } // end for jj

	         // new r
		 if (r+1!=1) {
		    // even case
		    if ((r+1)%2==0) {
		       // increase by 2 if possible
 		       if (r+1<2*m)
			  r+=2;
		       else // => r+1=2*m
			  r=2*m-2;
		    } // end if even case
		    else { // odd case
		       // decrease by 2 if possible
		       if (r+1>3)
			  r-=2;
		       else // => r+1=3
			  r=1;
		    }// end % if-else even/odd case
		 } // end if

	         // new s, same procedure
		 if (s+1!=1) {
		    // even case
		    if ((s+1)%2==0) {
		       // increase by 2 if possible
		       if (s+1<2*m)
		          s+=2;
		       else // => s+1=2*m
		          s=2*m-2;
		   } // end even case
		   else { // odd case
		       // decrease by 2 if possible
		       if (s+1>3)
		          s-=2;
		       else // => s+1=3
		          s=1;
		   } // end if-else even/odd case
		 } // end if

	         //  coordinate for next sweep
#pragma omp write
		 coord1[j]=r;
#pragma omp write
		 coord2[j]=s;
	     } // end for j
	     // end omp parallel for
	 } // end for i      
   } // end while
   hist[*iter]=nrm_offdgl/nrm_dgl;

   
   free(coord1);
   free(coord2);
   for (i=0; i<m; i++) {
       free(DD[i]);
       free(VV[i]);
       free(AA[i]);
   } // end for i
   free(DD);
   free(VV);
   free(AA);
   m=omp_get_max_threads();
   for (i=0; i<m; i++) {
       free(dbuff[i]);
       free(ind[i]);
       free(stack[i]);
   } 
   free(dbuff);
   free(ind);
   free(stack);

   // remove potential additional entries
   l=nn % (2*block_size);
   if (l) {
      m=2*block_size-l;
      double nrm2;
      r=0;
      s=n;
      j=0;
      pV=*V;
      pA=*A;
      ii=1;
      while (j<s) {
	    // nrm2 <-||V(1:nn,j)||_2
	    nrm2=dnrm2_(&nn,pV+j*n,&ii);
	    // this is an additional trivial eigenvector caused by augmentation
            if (nrm2<=1.0e-1) {
	       // shuffle last column to column j
	       s--;
	       r++;
	       memcpy(pV+j*n,pV+s*n,n*sizeof(double));
	       pA[j+j*n]=pA[s+s*n];
	       if (r==m)
                  break;
	    } // end if
	    else
	       j++;
      } // end while
      // now compress V and A to size nn x nn
      ii=m;
      pV=*V+n;
      pA=*A+n;
      for (j=1; j<nn; j++,ii+=m,pA+=n,pV+=n) {
	  memmove(pA-ii,pA,nn*sizeof(double));
	  memmove(pV-ii,pV,nn*sizeof(double));
      } // end for j
      // restore correct n
      n=nn;
   } // end if l

   
   // for clearness, sort eigenvalues and eigenvectors such that the
   // eigenvalues are ascending
   double *D   =(double *)malloc((size_t)n*sizeof(double));
   int *perm   =(int *)   malloc((size_t)n*sizeof(int));
   int *ibuffer=(int *)   malloc((size_t)n*sizeof(int));
   pA=*A;
   for (j=0; j<n; j++,pA+=n+1) {
       D[j]=*pA;
       perm[j]=j;
   } // end for j
   // sort eigenvalues in ascending order
   qsort_(D,perm,ibuffer,&n);
   // transfer sorted eigenvalues
   pA=*A;
   for (j=0; j<n; j++,pA+=n+1) 
       *pA=D[j];
   // copy associated permuted eigenvector to a buffer
   double *dbuffer=(double *)malloc((size_t)n*n*sizeof(double));
   pV=*V;
   for (j=0; j<n; j++)
       memcpy(dbuffer+j*n,pV+perm[j]*n,n*sizeof(double));
   // rewrite buffer
   memcpy(pV,dbuffer,n*n*sizeof(double));

   free(D);
   free(perm);
   free(ibuffer);
   free(dbuffer);
   

   if (*iter>=max_iter)
      return 1;
   else  
      return 0;
} // end jacobi
