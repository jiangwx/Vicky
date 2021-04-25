#include "CNN.h"

void pwconv(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l)
{

    for (int oh = 0; oh < l.oh; oh++)
    {
        for (int ow = 0; ow < l.ow; ow++)
        {
            for (int oc = 0; oc < l.oc; oc++)
            {
                float odata = 0;
                int ofm_index, weight_index, ifm_index;
                for (int ic = 0; ic < l.ic; ic++)
                {                   
                    ofm_index = (oc * l.ow * l.oh) + l.ow * oh + ow ;
                    weight_index = oc * l.ic + ic ;
                    ifm_index = (ic * l.ow * l.oh) + l.ow * oh + ow ;
                    odata = odata + weight[weight_index] * ifm [ifm_index];
                }
                odata += bias[oc];
                if(relu)
                {
                    if(odata>0)                
                        ofm[ofm_index] = odata;
                    else
                        ofm[ofm_index] = 0;
                }
                else
                {
                    ofm[ofm_index] = odata;
                }
                
            }
        }
    }
}

void dwconv(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l)
{
 
    for (int oh = 0; oh < l.oh; oh++)
    {
        for (int ow = 0; ow < l.ow; ow++)
        {
            for (int oc = 0; oc < l.oc; oc++)
            {
                float odata =0;
                float idata =0;
                int ofm_index, weight_index, ifm_index;
                for(int kh = 0; kh < l.k; kh++)
                {
                    for(int kw = 0; kw <l.k; kw++)
                    {
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        ofm_index = (oc * l.ow * l.oh) + l.ow * oh + ow ;
                        weight_index = oc * l.k * l.k + kh * l.k + kw ;
                        ifm_index = (oc * l.ow * l.oh) + l.ow * (oh + kh - l.p) + (ow + kw - l.p);

                        if ( (ow + kw - l.p) < 0 || (oh + kh - l.p) < 0 || (ow + kw - l.p) > l.ow - 1 || (oh + kh - l.p) > l.oh - 1 ) 
                            idata = 0;
                        else
                            idata = ifm [ifm_index];

                        odata += weight[weight_index] * idata;
                    }
                }
                odata += bias[oc];
                if(relu)
                {
                    if(odata>0)                
                        ofm[ofm_index] = odata;
                    else
                        ofm[ofm_index] = 0;
                }
                else
                {
                    ofm[ofm_index] = odata;
                }
            }
        }
    }
}

