#include "mc_s3.h"
#include "../tools/vendor/libs3/libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mc_s3 {
    s3_client *cl;
    char *url;
    mc_reader *r;
};

static int s3_read_cb(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    mc_s3 *s=ud;
    s3_response resp={0};
    if(s3_get_range(s->cl,s->url,off,len,&resp)!=S3_OK || !s3_response_ok(&resp)){
        s3_response_free(&resp);
        return -1;
    }
    int rc=-1;
    if(resp.status==206 && resp.body_len>=len){
        memcpy(dst,resp.body,len); rc=0;          // proper ranged reply
    } else if(resp.status==200 && resp.body_len>=off+len){
        memcpy(dst,resp.body+off,len); rc=0;      // server ignored Range and
    }                                             // sent the whole object
    s3_response_free(&resp);
    return rc;
}

mc_s3 *mc_s3_open(const char *url){
    if(!url) return NULL;
    mc_s3 *s=calloc(1,sizeof *s);
    s3_config cfg={0};
    s->cl=s3_client_new(&cfg);
    if(!s->cl){ free(s); return NULL; }
    s->url=strdup(url);
    s3_response head={0};
    uint64_t total=0;
    if(s3_head(s->cl,url,&head)==S3_OK && s3_response_ok(&head))
        total=head.content_length;
    s3_response_free(&head);
    if(!total){ mc_s3_close(s); return NULL; }
    s->r=mc_open_streaming(s3_read_cb,s,total);
    if(!s->r){ mc_s3_close(s); return NULL; }
    mc_reader_set_partial_fetch(s->r,1);
    return s;
}
mc_reader *mc_s3_reader(mc_s3 *s){ return s?s->r:NULL; }
void mc_s3_close(mc_s3 *s){
    if(!s) return;
    if(s->r) mc_close(s->r);
    if(s->cl) s3_client_free(s->cl);
    free(s->url);
    free(s);
}
