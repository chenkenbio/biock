#!/usr/bin/env python3

from biock.genomics import HG19_FASTA, HG19_CHROMSIZE
from biock.biock import run_bash
import warnings

AA_CODON = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}
codons = list(AA_CODON.keys())
for n1 in ['A', 'C', 'G', 'T', 'N']:
    for n2 in ['A', 'C', 'G', 'T', 'N']:
        for n3 in ['A', 'C', 'G', 'T', 'N']:
            codon = "{}{}{}".format(n1, n2, n3)
            if codon not in AA_CODON:
                AA_CODON[codon] = 'X'

for c in codons:
    AA_CODON[c.replace('T', 'U')] = AA_CODON[c]


NN_REVERSE = {
    "A": "T", "a": "t",
    "C": "G", "c": "g",
    "G": "C", "g": "c",
    "T": "A", "t": "a",
    "N": "N", "n": "n",
    '|': '|'
}


def reverse_seqs(seq):
    return ''.join([NN_REVERSE[n] for n in seq][::-1])



def interval_subtract(x1, x2, y1, y2):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    assert x1 <= x2 and y1 <= y2
    res = list()
    if y1 <= x1:
        if y2 <= x1:
            if x1 < x2:
                res.append((x1, x2))
        elif y2 > x1 and y2 < x2:
            res.append((y2, x2))
        elif y2 >= x2:
            pass
        else:
            raise RuntimeError()
    elif y1 > x1 and y1 < x2:
        res.append((x1, y1))
        if y2 < x2:
            res.append((y2, x2))
    elif y1 >= x2:
        pass
    else:
        raise RuntimeError()
    return res


class Transcript(object):
    def __init__(self, ucsc_tx_record):
        super().__init__()
        self.raw_record = ucsc_tx_record
        fields = ucsc_tx_record.strip().split('\t')
        self.tx_id, self.chrom, self.strand = fields[1:4]
        tx_start, tx_end, cds_start, cds_end = fields[4:8]
        self.tx_start, self.tx_end, self.cds_start, self.cds_end = \
            int(tx_start), int(tx_end), int(cds_start), int(cds_end)
        ## NOTE: cds_start/cds_end is independent to strand,
        ##       while TSS/TSE are related to the strand
        if self.strand == '+':
            self.tss, self.tse = self.tx_start, self.tx_end - 1
        else:
            self.tse, self.tss = self.tx_start, self.tx_end - 1
        exon_starts, exon_ends = fields[9:11]
        exon_starts = [int(x) for x in exon_starts.strip(',').split(',')]
        exon_ends = [int(x) for x in exon_ends.strip(',').split(',')]
        self.exons = list(zip(exon_starts, exon_ends))
        self.introns = [(self.exons[i][1], self.exons[i + 1][0]) for i in range(len(self.exons) - 1)]
        self.num_exons = len(self.exons)
        self.tx_length = abs(self.tx_end - self.tx_start)
        self.utr5, self.utr3 = list(), list()
        self.cds = list()
        for e1, e2 in self.exons:
            if e1 < self.cds_start:
                u1, u2 = e1, min(e2, self.cds_start)
                self.utr5.append((u1, u2))
                try:
                    cds = interval_subtract(e1, e2, u1, u2)
                except AssertionError as err:
                    print(ucsc_tx_record)
                    print(e1, e2, u1, u2)
                    raise AssertionError(err)
                if len(cds) > 0:
                    self.cds.extend(cds)
            if e2 > self.cds_end:
                u1, u2 = max(self.cds_end, e1), e2
                self.utr3.append((u1, u2))
                cds = interval_subtract(e1, e2, u1, u2)
                if len(cds) > 0:
                    self.cds.extend(cds)
            if e1 >= self.cds_start and e2 <= self.cds_end:
                self.cds.append((e1, e2))
        if self.strand == '-':
            self.utr5, self.utr3 = self.utr3, self.utr5
        self.premrna = None
        self.maturemrna = None
        self.coding_seq = None
        self.aa_seq = None

    
class AnnotatedTranscript(object):
    def __init__(self, tx: Transcript, flanking: int=2000, pad: int=0):
        super().__init__()
        """
        U: upstream
        5: UTR5
        C: cds
        I: intronic
        3: UTR3
        D downstream
        """
        self.raw_record = tx.raw_record
        self.tx_id = tx.tx_id
        self.chrom = tx.chrom
        self.tx_start = tx.tx_start
        self.tx_end = tx.tx_end
        self.cds_start= tx.cds_start
        self.cds_end= tx.cds_end
        self.strand = tx.strand
        if self.strand == '+':
            self.tss, self.tse = self.tx_start, self.tx_end - 1
        else:
            self.tss, self.tse = self.tx_end - 1, self.tx_start
        self.flanking = flanking
        self.exons = tx.exons.copy()
        self.introns = tx.introns.copy()
        self.left = self.tx_start - flanking
        self.right = self.tx_end + flanking
        self.forward_sequence = None
        self.forward_annotation = None
        self.forward_array = None
        self.forward_dist2ss = None
        self.forward_dist2tss = None
        self.__annotate_sequence(tx)
        assert self.forward_annotation.__len__() == self.forward_sequence.__len__()
    
    def get_sequence(self, chrom=None, start=None, end=None, sense=True, sep='|'):
        seq = sep.join(self.forward_sequence)
        if start is not None and end is not None:
            if sep != "":
                warnings.warn("sep: {} -> blank str".format(sep))
                seq = seq.replace(sep, '')
            assert chrom == self.chrom
            seq = seq[start - self.left: end - self.left]
        if sense and self.strand == '-':
            seq = reverse_seqs(seq)
        return seq
    
    def get_annotation(self, sense=True, sep='|'):
        anno = sep.join(self.forward_annotation)
        if sense and self.strand == '-':
            anno = anno[::-1]
        return anno
    
    def get_dist2ss(self, chrom=None, start=None, end=None, sense=True):
        if self.forward_dist2ss is None:
            dist2ss = list()
            ## upstream
            dist2ss.append(-np.arange(0, self.flanking)[::-1])
            for i, (e1, e2) in enumerate(self.exons):
                size = e2 - e1
                dist2ss.append(1 + np.concatenate((
                    np.arange(0, size).reshape(1, -1),
                    np.arange(size - 1, -1, -1).reshape(1, -1)
                ), axis=0).min(axis=0))
                if i < len(self.exons) - 1:
                    p1, p2 = self.introns[i]
                    size = p2 - p1
                    dist2ss.append(np.concatenate((
                        np.arange(0, size).reshape(1, -1),
                        np.arange(size - 1, -1, -1).reshape(1, -1)
                    ), axis=0).min(axis=0))
            dist2ss.append(-np.arange(0, self.flanking))
            self.forward_dist2ss = np.concatenate(dist2ss, axis=0)
        dist2ss = self.forward_dist2ss
        if start is not None and end is not None:
            assert chrom == self.chrom
            assert start >= self.left and end <= self.right, "{}: [{}, {}) is valid, while [start, end) = [{}, {})".format(self.tx_id, self.left, self.right, start, end)
            dist2ss = dist2ss[start - self.left: end - self.left]
        if sense and self.strand == '-':
            dist2ss = dist2ss[::-1]
        return dist2ss
    
    def __annotate_sequence(self, tx: Transcript):
        """
        use forward strand internally
        """
        ANNO_DICT = {"upstream": 'U', "UTR5": '5', "CDS": "C", "intron": 'I', "UTR3": '3', "downstream": 'D'}
        left_pad = '_' * max(0, self.flanking - self.tx_start)
        right_pad = '_' * max(self.tx_end + self.flanking - HG19_CHROMSIZE[tx.chrom], 0)
        rc, out, err = run_bash("samtools faidx {} {}:{}-{}".format(
            HG19_FASTA, tx.chrom, 
            1 + max(0, tx.tx_start - self.flanking), 
            min(HG19_CHROMSIZE[self.chrom],  tx.tx_end + self.flanking)))
        if rc != 0:
            raise RuntimeError(err)
        forward_sequence = "{}{}{}".format(
            left_pad,
            ''.join(out.strip().split('\n')[1:]),
            right_pad
        )
        # print(self.forward_sequence[0:100])
        
        self.forward_sequence = list()
        self.forward_annotation = list()
        self.regions = list() # (region, start, end)
        self.regions.append((
            self.tx_start - self.flanking - self.left, 
            self.tx_start - self.left, 
            "upstream" if tx.strand == '+' else "downstream"
        ))
        x, y, r = self.regions[-1]
        self.forward_annotation.append(ANNO_DICT[r] * (y - x))
        for i, (e1, e2) in enumerate(tx.exons):
            u1, u2 = max(e1, self.tx_start), min(e2, self.cds_start)
            if u1 < u2:
                self.regions.append((
                    u1 - self.left, u2 - self.left,
                    "UTR5" if tx.strand == '+' else "UTR3"
                ))
                x, y, r = self.regions[-1]
                self.forward_annotation.append(ANNO_DICT[r] * (y - x))

            c1, c2 = max(e1, self.cds_start), min(e2, self.cds_end)
            if c1 < c2:
                self.regions.append((
                    c1 - self.left, 
                    c2 - self.left,
                    "CDS"
                ))
                x, y, r = self.regions[-1]
                self.forward_annotation.append(ANNO_DICT[r] * (y - x))

            u1, u2 = max(e1, self.cds_end), min(e2, self.tx_end)
            if u1 < u2:
                self.regions.append((
                    u1 - self.left, u2 - self.left,
                    "UTR3" if tx.strand == '+' else "UTR5"
                ))
                x, y, r = self.regions[-1]
                self.forward_annotation.append(ANNO_DICT[r] * (y - x))

            if i < tx.num_exons - 1:
                p1, p2 = tx.introns[i]  
                self.regions.append((
                    p1 - self.left, p2 - self.left,
                    "intron"
                ))
                x, y, r = self.regions[-1]
                self.forward_annotation.append(ANNO_DICT[r] * (y - x))

        self.regions.append((
            self.tx_end - self.left, 
            self.tx_end + self.flanking - self.left,
            "downstream" if tx.strand == '+' else "upstream"
        ))
        x, y, r = self.regions[-1]
        self.forward_annotation.append(ANNO_DICT[r] * (y - x))

        total_len = 0
        for x, y, r in self.regions:
            total_len += (y - x)
            if r == "CDS" or r == "UTR3" or r == "UTR5":
                self.forward_sequence.append(forward_sequence[x:y].upper())
            else:
                self.forward_sequence.append(forward_sequence[x:y].lower())
        assert total_len == self.tx_end - self.tx_start + 2 * self.flanking

    # def get_mutant_sequence(self, chrom, ref_start, ref_end, ref, alt):
    #     assert len(ref) == ref_end - ref_start, "The coordinates should be [ref_start, ref_end), 0-start"
    #     assert chrom == self.chrom
    #     before, ref_list, after = list(), list(), list()
    #     ref_start, ref_end = ref_start - self.left, ref_end - self.left
    #     forward_seq = ''.join(self.forward_sequence)
    #     assert forward_seq[ref_start:ref_end].upper() == ref.upper(), "\nexpected:{}\nfound:   {}\n".format(forward_seq[ref_start:ref_end], ref)
    #     for start, stop, region in self.regions:
    #         if stop <= ref_start: # [start < stop) <= ref_start < ref_end
    #             before.extend([forward_seq[start:stop], '|'])
    #         elif start < ref_start: # start < ref_start < stop 
    #             before.append(forward_seq[start:ref_start])
    #             if stop < ref_end:  # [start < ref_start < stop) < ref_end
    #                 ref_list.extend([forward_seq[ref_start:stop], '|'])
    #             else: # start < ref_start < ref_end <= stop
    #                 ref_list.append(forward_seq[ref_start:ref_end])
    #                 if ref_end < stop:
    #                     after.append(forward_seq[ref_end:stop])
    #                 after.append('|')
    #         elif start < ref_end: # ref_start <= start < ref_end
    #             if stop < ref_end: # ref_start <= start < stop < ref_end
    #                 ref_list.extend([forward_seq[start:stop], '|'])
    #             else: # ref_start <= start < ref_end <= stop
    #                 ref_list.append(forward_seq[start:ref_end])
    #                 if ref_end < stop:
    #                     after.append(forward_seq[ref_end:stop])
    #                 after.append('|')
    #         else: # start >= ref_end stop > ref_start
    #             after.extend([forward_seq[start:stop], '|'])

    #     before = ''.join(before)
    #     ref_list = ''.join(ref_list)
    #     after = ''.join(after).rstrip('|')
    #     if self.strand == '+':
    #         s = "{}[{}/{}]{}".format(before, ref_list, alt, after)
    #     else:
    #         s = "{}[{}/{}]{}".format(reverse_seqs(after), reverse_seqs(ref_list), reverse_seqs(alt), reverse_seqs(before))
    #     assert before.replace('|', '').upper() + ref.upper() + after.replace('|', '').upper() == forward_seq.upper()
    #     return s

    def translate(self):
        # splice site mutation
        forward_seq = ''.join(self.forward_sequence)
        cds = list()
        for c1, c2, r in self.regions:
            if r == "CDS":
                cds.append(forward_seq[c1:c2])
        cds = ''.join(cds)
        if self.strand ==  '-':
            cds = reverse_seqs(cds)
        n_aa = len(cds) // 3
        aa_seq = [AA_CODON[cds[i * 3: (i + 1) * 3].upper()] for i in range(n_aa)]
        aa_seq = ''.join(aa_seq)
        if len(aa_seq) == 0:
            pass
        elif aa_seq[-1] != '*':
            warnings.warn("Missing codon for {}".format(self.tx_id))
        elif aa_seq.count('*') > 1:
            warnings.warn("Multiple stop codon for {}".format(self.tx_id))
        
        return aa_seq
    
    def __str__(self):
        if self.strand == '+':
            s = "{}\n{}_{}:{}-{}_{}\n{}".format(
                self.raw_record, self.tx_id, self.chrom, self.tx_start, self.tx_end, self.strand,
                '\n'.join(["{}:\t[ {} , {} ) / [{} , {} )".format(r, x1 + self.left, x2 + self.left, x1 + self.left - self.tx_start, x2 + self.left - self.tx_start) for x1, x2, r in self.regions])
            )
        else:
            s = "{}\n{}_{}:{}-{}_{}\n{}".format(
                self.raw_record, self.tx_id, self.chrom, self.tx_start, self.tx_end, self.strand,
                '\n'.join(["{}:\t[ {} , {} ) / [{} , {} )".format(r, x1 + self.left, x2 + self.left, self.tx_end - x1 - self.left, self.tx_end - x2 - self.left) for x1, x2, r in self.regions])
            )
        return s


def load_transcripts(fn="/home/chenken/db/annovar/humandb/hg19_refGene.txt"):
    # cache = os.path.join(processed_path, "{}_cache.pt".format(os.path.basename(fn)))
    # if os.path.exists(cache):
    #     transcripts = torch.load(cache)
    # else:
    transcripts = dict()
    with open(fn) as infile:
        # for l in tqdm.tqdm(infile, desc="Loading transcripts"):
        for l in infile:
            t = Transcript(l.strip())
            transcripts["{}@{}".format(t.tx_id, t.chrom)] = t
    #     torch.save(transcripts, cache)
    return transcripts



# class Transcript(object):
#     """
#     exon: 1-based, close intervals
#     """
#     def __init__(self, tx_id, gene_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, exons):
#         self.tx_id = tx_id
#         self.gene_id = gene_id
#         self.chrom = chrom
#         self.strand = strand
#         self.tx_start = int(tx_start)
#         self.tx_end = int(tx_end)
#         self.cds_start = int(cds_start)
#         self.cds_end = int(cds_end)
#         self.exons = exons
#         self.introns = self.get_introns(self.exons)
#         self.tx_length = sum([b - a + 1 for a, b in self.exons])
# 
#     def get_introns(self, exons):
#         introns = list()
#         for i in range(len(exons) - 1):
#             introns.append((exons[i][1] + 1, exons[i + 1][0] - 1))
#         return introns.copy()
#     def __str__(self):
#         cnt = 0
#         s = "{} {}:{}-{} {} CDNA:{}bp\n".format(self.tx_id, self.chrom, self.tx_start, self.tx_end, self.strand, self.tx_length)
#         for exs, exe in self.exons:
#             s += "{} - {}\t{} - {} | {} - {}\n".format(
#                     exs, exe, 
#                     cnt + 1, exe - exs + 1 + cnt, 
#                     self.tx_length - cnt, self.tx_length - cnt - exe + exs
#                 )
#             cnt += exe - exs + 1
#         return s
# 
# 
#     def genome2transcript(self, position, **kwargs):
#         """
#         position: 1-based
#         """
#         if 'chrom' in kwargs:
#             assert self.chrom.lstrip('chr') == kwargs['chrom']
#         if 'chr' in kwargs:
#             assert self.chrom.lstrip('chr') == kwargs['chrom']
#         region, cdna_pos = None, None
#         if position < self.exons[0][0]:
#             region, cdna_pos = "upstream", "{}-{}".format("", abs(position - self.exons[0][0]))
#         elif position > self.exons[-1][1]:
#             region, cdna_pos = "downstream", "{}+{}".format("", position - self.exons[-1][1])
#         else:
#             cdna_cnt = 0
#             for i, (exs, exe) in enumerate(self.exons):
#                 if position >= exs and position <= exe:
#                     region = "exon{}".format(i + 1)
#                     cdna_pos = cdna_cnt + position - exs + 1
#                     break
#                 else:
#                     cdna_cnt += exe - exs + 1
#             if cdna_pos is None:
#                 cdna_cnt = 0
#                 for i, (ins, ine) in enumerate(self.introns):
#                     cdna_cnt += self.exons[i][1] - self.exons[i][0] + 1
#                     if position >= ins and position <= ine:
#                         region = "intron{}".format(i + 1)
#                         if position - ins < ine - position:
#                             cdna_pos = "{}+{}".format(cdna_cnt, position - ins + 1)
#                         else:
#                             cdna_pos = "{}-{}".format(cdna_cnt + 1, self.introns[i][1] - position + 1)
#         if self.strand == '-':
#             if region == "upstream":
#                 region = "downstream"
#                 cdna_pos = cdna_pos.replace('-', '+')
#             elif region.startswith('exon'):
#                 region = "exon{}".format(1 + len(self.exons) - int(region.replace('exon', '')))
#                 cdna_pos = self.tx_length + 1 - cdna_pos
#             elif region.startswith('intron'):
#                 region = "intron{}".format(1 + len(self.introns) - int(region.replace('intron', '')))
#                 if '+' in cdna_pos:
#                     left_cnt, pos = cdna_pos.split('+')
#                     cdna_pos = "{}-{}".format(self.tx_length - int(left_cnt) + 1, pos)
#                 else:
#                     left_cnt, pos = cdna_pos.split('-')
#                     cdna_pos = "{}+{}".format(self.tx_length - (int(left_cnt) - 1), pos)
#             elif region == "downstream":
#                 region == "upstream"
#                 cdna_pos = cdna_pos.replace('+', '-')
#         return region, cdna_pos
# 
# 
# class TranscriptDB(object):
#     def __init__(self, db, dbtype='annovar'):
#         self.transcripts = dict()
#         if dbtype == "annovar":
#             self.load_from_annovar(db)
#         else:
#             raise NotImplementedError
# 
#     def load_from_annovar(self, db):
#         with open(db) as infile:
#             for l in infile:
#                 _, tx_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, _, exon_starts, exon_ends, _, gene_id, _, _, _ = l.strip('\n').split('\t')
#                 tx_start, tx_end, cds_start, cds_end = int(tx_start), int(tx_end), int(cds_start), int(cds_end)
#                 exons = list(zip(
#                         [int(x) + 1 for x in exon_starts.strip(',').split(',')],
#                         [int(x) for x in exon_ends.strip(',').split(',')]
#                         ))
#                 t = Transcript(tx_id, gene_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, exons)
#                 self.transcripts[t.tx_id] = t
# 
# 
# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(args.seed)
# 
#     txdb = TranscriptDB(db="/home/chenken/db/annovar/humandb/hg19_refGene.txt")
# 
#     # NM_001160184 chr1:901861-911245 + CDNA:3071bp
#     # 901862 - 901994	1 - 133 | 3071 - 2939
#     # 902084 - 902183	134 - 233 | 2938 - 2839
#     # 905657 - 905803	234 - 380 | 2838 - 2692
#     # 905901 - 905981	381 - 461 | 2691 - 2611
#     # 906066 - 906138	462 - 534 | 2610 - 2538
#     # 906259 - 906386	535 - 662 | 2537 - 2410
#     # 906457 - 906588	663 - 794 | 2409 - 2278
#     # 906704 - 906784	795 - 875 | 2277 - 2197
#     # 907455 - 907530	876 - 951 | 2196 - 2121
#     # 907668 - 907804	952 - 1088 | 2120 - 1984
#     # 908241 - 908390	1089 - 1238 | 1983 - 1834
#     # 908880 - 909020	1239 - 1379 | 1833 - 1693
#     # 909213 - 909431	1380 - 1598 | 1692 - 1474
#     # 909696 - 909744	1599 - 1647 | 1473 - 1425
#     # 909822 - 911245	1648 - 3071 | 1424 - 1
#     print()
#     t = txdb.transcripts['NM_001160184']
#     print(t.genome2transcript(901862) == ("exon1", 1))
#     print(t.genome2transcript(901994) == ("exon1", 133))
#     print(t.genome2transcript(901995) == ("intron1", "133+1"))
# 
#     # NM_001349996 chr22:36134782-36357678 - CDNA:6692bp exon:14
#     # 36134783 - 36140296	1 - 5514 | 6692 - 1179
#     # 36141970 - 36142042	5515 - 5587 | 1178 - 1106
#     # 36142520 - 36142608	5588 - 5676 | 1105 - 1017
#     # 36152152 - 36152191	5677 - 5716 | 1016 - 977
#     # 36155935 - 36156067	5717 - 5849 | 976 - 844
#     # 36157249 - 36157341	5850 - 5942 | 843 - 751
#     # 36157462 - 36157515	5943 - 5996 | 750 - 697
#     # 36161470 - 36161530	5997 - 6057 | 696 - 636
#     # 36164304 - 36164396	6058 - 6150 | 635 - 543
#     # 36174072 - 36174122	6151 - 6201 | 542 - 492
#     # 36177647 - 36177793	6202 - 6348 | 491 - 345
#     # 36205827 - 36206051	6349 - 6573 | 344 - 120
#     # 36334895 - 36334945	6574 - 6624 | 119 - 69
#     # 36357611 - 36357678	6625 - 6692 | 68 - 1
# 
#     print()
#     t = txdb.transcripts['NM_001349996']
#     print(t.genome2transcript(36134782) == ("downstream", "+1"))
#     print(t.genome2transcript(36134783) == ("exon14", 6692))
#     print(t.genome2transcript(36140297) == ("intron13", "1179-1"))
#     print(t.genome2transcript(36141968) == ("intron13", "1178+2"))
#     print(t.genome2transcript(36357678) == ("exon1", 1))



