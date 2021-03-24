#!/usr/bin/env python3


class Transcript(object):
    """
    exon: 1-based, close intervals
    """
    def __init__(self, tx_id, gene_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, exons):
        self.tx_id = tx_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.tx_start = int(tx_start)
        self.tx_end = int(tx_end)
        self.cds_start = int(cds_start)
        self.cds_end = int(cds_end)
        self.exons = exons
        self.introns = self.get_introns(self.exons)
        self.tx_length = sum([b - a + 1 for a, b in self.exons])

    def get_introns(self, exons):
        introns = list()
        for i in range(len(exons) - 1):
            introns.append((exons[i][1] + 1, exons[i + 1][0] - 1))
        return introns.copy()
    def __str__(self):
        cnt = 0
        s = "{} {}:{}-{} {} CDNA:{}bp\n".format(self.tx_id, self.chrom, self.tx_start, self.tx_end, self.strand, self.tx_length)
        for exs, exe in self.exons:
            s += "{} - {}\t{} - {} | {} - {}\n".format(
                    exs, exe, 
                    cnt + 1, exe - exs + 1 + cnt, 
                    self.tx_length - cnt, self.tx_length - cnt - exe + exs
                )
            cnt += exe - exs + 1
        return s


    def genome2transcript(self, position, **kwargs):
        """
        position: 1-based
        """
        if 'chrom' in kwargs:
            assert self.chrom.lstrip('chr') == kwargs['chrom']
        if 'chr' in kwargs:
            assert self.chrom.lstrip('chr') == kwargs['chrom']
        region, cdna_pos = None, None
        if position < self.exons[0][0]:
            region, cdna_pos = "upstream", "{}-{}".format("", abs(position - self.exons[0][0]))
        elif position > self.exons[-1][1]:
            region, cdna_pos = "downstream", "{}+{}".format("", position - self.exons[-1][1])
        else:
            cdna_cnt = 0
            for i, (exs, exe) in enumerate(self.exons):
                if position >= exs and position <= exe:
                    region = "exon{}".format(i + 1)
                    cdna_pos = cdna_cnt + position - exs + 1
                    break
                else:
                    cdna_cnt += exe - exs + 1
            if cdna_pos is None:
                cdna_cnt = 0
                for i, (ins, ine) in enumerate(self.introns):
                    cdna_cnt += self.exons[i][1] - self.exons[i][0] + 1
                    if position >= ins and position <= ine:
                        region = "intron{}".format(i + 1)
                        if position - ins < ine - position:
                            cdna_pos = "{}+{}".format(cdna_cnt, position - ins + 1)
                        else:
                            cdna_pos = "{}-{}".format(cdna_cnt + 1, self.introns[i][1] - position + 1)
        if self.strand == '-':
            if region == "upstream":
                region = "downstream"
                cdna_pos = cdna_pos.replace('-', '+')
            elif region.startswith('exon'):
                region = "exon{}".format(1 + len(self.exons) - int(region.replace('exon', '')))
                cdna_pos = self.tx_length + 1 - cdna_pos
            elif region.startswith('intron'):
                region = "intron{}".format(1 + len(self.introns) - int(region.replace('intron', '')))
                if '+' in cdna_pos:
                    left_cnt, pos = cdna_pos.split('+')
                    cdna_pos = "{}-{}".format(self.tx_length - int(left_cnt) + 1, pos)
                else:
                    left_cnt, pos = cdna_pos.split('-')
                    cdna_pos = "{}+{}".format(self.tx_length - (int(left_cnt) - 1), pos)
            elif region == "downstream":
                region == "upstream"
                cdna_pos = cdna_pos.replace('+', '-')
        return region, cdna_pos


class TranscriptDB(object):
    def __init__(self, db, dbtype='annovar'):
        self.transcripts = dict()
        if dbtype == "annovar":
            self.load_from_annovar(db)
        else:
            raise NotImplementedError

    def load_from_annovar(self, db):
        with open(db) as infile:
            for l in infile:
                _, tx_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, _, exon_starts, exon_ends, _, gene_id, _, _, _ = l.strip('\n').split('\t')
                tx_start, tx_end, cds_start, cds_end = int(tx_start), int(tx_end), int(cds_start), int(cds_end)
                exons = list(zip(
                        [int(x) + 1 for x in exon_starts.strip(',').split(',')],
                        [int(x) for x in exon_ends.strip(',').split(',')]
                        ))
                t = Transcript(tx_id, gene_id, chrom, strand, tx_start, tx_end, cds_start, cds_end, exons)
                self.transcripts[t.tx_id] = t


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)

    txdb = TranscriptDB(db="/home/chenken/db/annovar/humandb/hg19_refGene.txt")

    # NM_001160184 chr1:901861-911245 + CDNA:3071bp
    # 901862 - 901994	1 - 133 | 3071 - 2939
    # 902084 - 902183	134 - 233 | 2938 - 2839
    # 905657 - 905803	234 - 380 | 2838 - 2692
    # 905901 - 905981	381 - 461 | 2691 - 2611
    # 906066 - 906138	462 - 534 | 2610 - 2538
    # 906259 - 906386	535 - 662 | 2537 - 2410
    # 906457 - 906588	663 - 794 | 2409 - 2278
    # 906704 - 906784	795 - 875 | 2277 - 2197
    # 907455 - 907530	876 - 951 | 2196 - 2121
    # 907668 - 907804	952 - 1088 | 2120 - 1984
    # 908241 - 908390	1089 - 1238 | 1983 - 1834
    # 908880 - 909020	1239 - 1379 | 1833 - 1693
    # 909213 - 909431	1380 - 1598 | 1692 - 1474
    # 909696 - 909744	1599 - 1647 | 1473 - 1425
    # 909822 - 911245	1648 - 3071 | 1424 - 1
    print()
    t = txdb.transcripts['NM_001160184']
    print(t.genome2transcript(901862) == ("exon1", 1))
    print(t.genome2transcript(901994) == ("exon1", 133))
    print(t.genome2transcript(901995) == ("intron1", "133+1"))

    # NM_001349996 chr22:36134782-36357678 - CDNA:6692bp exon:14
    # 36134783 - 36140296	1 - 5514 | 6692 - 1179
    # 36141970 - 36142042	5515 - 5587 | 1178 - 1106
    # 36142520 - 36142608	5588 - 5676 | 1105 - 1017
    # 36152152 - 36152191	5677 - 5716 | 1016 - 977
    # 36155935 - 36156067	5717 - 5849 | 976 - 844
    # 36157249 - 36157341	5850 - 5942 | 843 - 751
    # 36157462 - 36157515	5943 - 5996 | 750 - 697
    # 36161470 - 36161530	5997 - 6057 | 696 - 636
    # 36164304 - 36164396	6058 - 6150 | 635 - 543
    # 36174072 - 36174122	6151 - 6201 | 542 - 492
    # 36177647 - 36177793	6202 - 6348 | 491 - 345
    # 36205827 - 36206051	6349 - 6573 | 344 - 120
    # 36334895 - 36334945	6574 - 6624 | 119 - 69
    # 36357611 - 36357678	6625 - 6692 | 68 - 1

    print()
    t = txdb.transcripts['NM_001349996']
    print(t.genome2transcript(36134782) == ("downstream", "+1"))
    print(t.genome2transcript(36134783) == ("exon14", 6692))
    print(t.genome2transcript(36140297) == ("intron13", "1179-1"))
    print(t.genome2transcript(36141968) == ("intron13", "1178+2"))
    print(t.genome2transcript(36357678) == ("exon1", 1))
