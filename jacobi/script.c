#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define MAX_ITER 1000


int jacobi(double **A, int n, double **V, int block_size, double tol, double *hist, int *iter);







int main(int argc, char **argv)
{
   FILE *fp;
   int n, i,j;
   double *A, *pA;
   
   // read in matrix in matrix market format
   if (argc!=3) {
      fprintf(stderr, "Usage: %s [block size] [number of threads]\n", argv[0]);
      exit(1);
   }
   else { 
      if ((fp=fopen("symmetric_matrix.mat", "r"))==NULL) 
	 exit(1);
   }
   int block_size=atoi(argv[1]);
   omp_set_num_threads(atoi(argv[2]));
   
   fscanf(fp,"%d",&n);
   A=(double *)malloc((size_t)n*n*sizeof(double));
   pA=A;
   for (j=0; j<n; j++) 
       for (i=0; i<n; i++,pA++)
           fscanf(fp,"%le",pA);
   fclose(fp);

   if (n<=10) {
      printf("Matrix A on input, size %d x %d\n", n,n);
      for (i=0; i<n; i++,pA++) {
          pA=A+i;
	  for (j=0; j<n; j++,pA+=n) 
              printf("%12.4le",*pA);
	  printf("\n");
      }
      printf("\n"); fflush(stdout);
   }


   double *V=(double *)malloc((size_t)n*n*sizeof(double)), *pV;
   int iter=MAX_ITER;
   double *hist=(double *)malloc((size_t)(iter+1)*sizeof(double));

   double wtime=omp_get_wtime();
   int ierr=jacobi(&A,n,&V, block_size, 1e-12, hist, &iter);
   printf("computation time %8.1le [sec]\n",omp_get_wtime()-wtime);
   
   if (ierr)
      printf("Jacobi did not converge within %d sweeps\n", iter);
   else
      printf("Jacobi successfully converged after %d sweeps\n", iter);

   if (n<=10) {
      printf("Matrix A on output, size %d x %d\n", n,n);
      for (i=0; i<n; i++) {
          pA=A+i;
	  for (j=0; j<n; j++,pA+=n) 
              printf("%12.4le",*pA);
	  printf("\n");
      }
      printf("\n"); fflush(stdout);

      printf("Matrix V on output, size %d x %d\n", n,n);
      for (i=0; i<n; i++) {
	  pV=V+i;
	  for (j=0; j<n; j++,pV+=n) 
              printf("%12.4le",*pV);
	  printf("\n");
      }
      printf("\n"); fflush(stdout);
   } // end if
   
   fp=fopen("eigenvalues","w");
   fprintf(fp,"%d\n",n);
   pA=A;
   for (i=0; i<n; i++,pA+=n+1)
       fprintf(fp,"%24.16le\n",*pA);
   fclose(fp);

   fp=fopen("history","w");
   fprintf(fp,"%d\n",iter+1);
   for (i=0; i<=iter; i++)
       fprintf(fp,"%8.1le\n",hist[i]);
   fclose(fp);
   
   free(A);
   free(V);
   free(hist);
   
   return (0);
} // end main

