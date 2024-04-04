/*
 * SMART: string matching algorithms research tool.
 * Copyright (C) 2012  Simone Faro and Thierry Lecroq
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 * 
 * contact the authors at: faro@dmi.unict.it, thierry.lecroq@univ-rouen.fr
 * download the tool at: http://www.dmi.unict.it/~faro/smart/
 */

#include "timer.h"
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BEGIN_PREPROCESSING	{timer_start(_timer);}
#define BEGIN_SEARCHING		{timer_start(_timer);}
#define END_PREPROCESSING	{timer_stop(_timer);pre_time = timer_elapsed(_timer)*1000;}
#define END_SEARCHING		{timer_stop(_timer);run_time = timer_elapsed(_timer)*1000;}

/* global variables used for computing preprocessing and searching times */
double run_time, 		// searching time
	   pre_time;	// preprocessing time
TIMER * _timer;

int search(unsigned char* p, int m, unsigned char* t, int n);

long read_file_content( unsigned char**buffer, const char* filename ) {
	long length = -1;
	FILE * file = fopen (filename, "rb");
	if (file) {
		fseek (file, 0, SEEK_END);
		length = ftell (file);
		fseek (file, 0, SEEK_SET);
		*buffer = (unsigned char*)malloc (length+1);
		if (*buffer==NULL) return 0;
		if(fread (*buffer, 1, length, file)!=length) { printf("Error (read_file_content): Something went wrong"); exit(EXIT_FAILURE); }
		*(*buffer+length) = '\0';
		fclose (file);
	} else {
		printf("Error (read_file_content): File (%s) do not exists or you don't have permission", filename); exit(EXIT_FAILURE);
	}
	return length;
}

int main(int argc, char *argv[])
{
	_timer = (TIMER*) malloc (sizeof(TIMER));
    unsigned char *p, *filename;
	int m, n, text_size;
	unsigned char *t = NULL;
	if(argc < 5) {
		printf("error in input parameter\nfour parameters needed in standard mode\n");
		return 1;
	}
	p = (unsigned char*) argv[1];
	m = atoi(argv[2]);
	filename = (unsigned char*) argv[3];
	n = atoi(argv[4]);
	
	text_size = read_file_content(&t, filename);

	int occ = search(p,m,t,n);

	printf("%s\t%d\t%s\t%d\t%f\t%f\t%d\n", p, m, filename, n, pre_time, run_time, occ); fflush(stdout);
	return 0;
}
