#include "CNN.h"

void reorg(float *ifm, float *ofm, layer l)
{
    for(int c=0; c<l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = c*l.ih*l.iw + (2*h)*l.iw + (2*w);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=l.ic; c<2*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-l.ic)*l.ih*l.iw + (2*h)*l.iw + (2*w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=2*l.ic; c<3*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-2*l.ic)*l.ih*l.iw + (2*h+1)*l.iw + (2*w);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=3*l.ic; c<4*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-3*l.ic)*l.ih*l.iw + (2*h+1)*l.iw + (2*w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
}