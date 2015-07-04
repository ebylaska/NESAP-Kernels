/*$Id: util_date.c,v 1.8 2003-08-13 18:06:11 edo Exp $*/

#include <sys/types.h>
#include <time.h>
#include	<string.h>

void util_date_(char *date, int flen)
{
  int i;
  time_t t = time((time_t *) 0);
  char *tmp = ctime(&t);
  int nlen = strlen(tmp);
  tmp[nlen-1] = '\0';
  for (i=0; i<nlen; ++i)
     date[i] = tmp[i];
  for (i=nlen; i<flen; ++i)
     date[i] = tmp[i];
}




