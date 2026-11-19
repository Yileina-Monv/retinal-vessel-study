"""CPU-only analysis of one seed; same frozen metric/aggregation implementation."""
import os,argparse
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[2027,2028],required=True);p.add_argument('--fixture',action='store_true');a=p.parse_args()
    os.environ.update(RETINA_NN_SEED=str(a.seed),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import replication_gate as g
    from replication_analysis import analyze
    if a.fixture:analyze(out=g.STAGE/'cpu_reference/full_endpoint_fixture',fixture=True)
    else:analyze()
