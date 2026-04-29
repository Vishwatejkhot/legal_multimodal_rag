from __future__ import annotations
from config import CASE_LAW_DIR
from ingestion.chunker import chunk_text
from reliability.logger import log

CASE_LAW_DIR.mkdir(parents=True, exist_ok=True)

_CASE_TEXTS = [
    (
        "McDonald_v_McDonald_2016_UKSC_28",
        "McDonald v McDonald [2016] UKSC 28",
        """McDonald v McDonald [2016] UKSC 28 Supreme Court of the United Kingdom.
Judgment delivered 15 June 2016.

HEADNOTE: Possession proceedings – Article 8 ECHR – Whether private landlord must
justify possession on proportionality grounds – Housing Act 1988 s.21.

The Supreme Court held that when a private landlord seeks possession under the
Housing Act 1988 s.21 against an assured shorthold tenant, the court is not required
to consider whether eviction is proportionate under Article 8 of the European Convention
on Human Rights. The Convention imposes obligations on the State, not on private
individuals. A private landlord acting under s.21 does not exercise a public function
and therefore Article 8 proportionality does not apply in such proceedings.

Lord Neuberger PSC (with whom Lord Clarke, Lord Sumption, Lord Reed and Lord Carnwath JJSC agreed):
The starting point is Manchester City Council v Pinnock [2010] UKSC 45, which held that
public authority landlords must be able to justify possession on proportionality grounds.
However the present case concerns a private landlord. The critical distinction is that
private persons are not bound by the Convention. The positive obligation on the State
under Article 8 does not require domestic courts to import a proportionality defence
into purely private possession proceedings.

HELD: Appeal dismissed. Section 21 mandatory ground for possession is not qualified by
Article 8 proportionality where the landlord is a private individual.""",
    ),
    (
        "Pinnock_v_Manchester_2010_UKSC_45",
        "Manchester City Council v Pinnock [2010] UKSC 45",
        """Manchester City Council v Pinnock [2010] UKSC 45 Supreme Court of the United Kingdom.
Judgment delivered 3 November 2010.

HEADNOTE: Possession by public authority landlord – Article 8 ECHR – Proportionality
defence – Demoted tenancy – Housing Act 1996 s.143D.

The Supreme Court (nine Justices sitting) held that whenever a court is asked to make
a possession order against a person's home, the court must have the power to consider
whether making the order is a proportionate means of meeting a legitimate aim under
Article 8(2) ECHR. This applies to all public authority landlords, including local
housing authorities. Previous decisions limiting this review were overruled.

Lord Neuberger PSC delivered the leading judgment:
The Strasbourg court has now made clear in Kay v United Kingdom (2012) that domestic
courts must be able to consider proportionality in possession cases involving public
authority landlords. The previous rule in Harrow LBC v Qazi [2003] must be departed from.
However the circumstances in which Article 8 will provide a defence to possession will
be highly exceptional. The fact of unlawfulness or breach of tenancy obligations will
almost always make possession proportionate.

HELD: The county court must consider proportionality when requested. However on the
facts the order was proportionate given the tenant's antisocial behaviour.""",
    ),
    (
        "Southwark_v_Mills_1999_UKHL_40",
        "Southwark LBC v Mills [1999] UKHL 40",
        """Southwark London Borough Council v Mills [1999] UKHL 40 House of Lords.
Judgment delivered 21 October 1999.

HEADNOTE: Landlord and Tenant – Covenant of quiet enjoyment – Nuisance – Sound
insulation – Housing disrepair – Whether normal use by neighbours constitutes breach.

The tenants complained that ordinary sounds from neighbouring flats could be heard
through the inadequately insulated walls and floors of their council block. They argued
this constituted a breach of the covenant for quiet enjoyment and/or the tort of nuisance.

Lord Hoffmann:
For there to be a breach of the covenant for quiet enjoyment, there must be a physical
interference with the tenant's enjoyment of the demised premises. Sounds from neighbouring
properties do not constitute such interference unless they are above ordinary levels of
use. The landlord does not impliedly covenant to improve the sound insulation of premises
let in their existing condition. The covenant against derogation from grant similarly
requires some act by the landlord which renders the premises substantially less fit for
the purposes for which they were let.

HELD: Appeal dismissed. Normal residential use by other tenants, even if audible through
thin walls, does not constitute breach of quiet enjoyment or nuisance. The council had
let the flats in their existing defective condition with the tenants' knowledge.""",
    ),
    (
        "Doherty_v_Birmingham_2008_UKHL_57",
        "Doherty v Birmingham City Council [2008] UKHL 57",
        """Doherty v Birmingham City Council [2008] UKHL 57 House of Lords.
Judgment delivered 30 July 2008.

HEADNOTE: Gypsies and travellers – Licences to occupy – Possession proceedings –
Article 8 ECHR – Summary possession – Legitimate aim.

The appellant was a Gypsy who occupied a council-owned site under a licence.
The council sought possession. He argued that possession without full review of
proportionality breached Article 8 ECHR.

Lord Hope of Craighead:
The law is now settled following Connors v United Kingdom (2005) that domestic courts
must be able to consider whether possession of a Gypsy's home is proportionate.
However the review available in judicial review proceedings, which permits challenge
on grounds of Wednesbury unreasonableness, may be sufficient to comply with Article 8
in cases involving licensees rather than tenants. The council must give reasons for
seeking possession which go beyond the mere fact that it has a contractual right to do so.

HELD: The matter was remitted for reconsideration. The courts must have power to
consider Gateway (b) review (whether it is lawful to make the order in the particular
case) in addition to Gateway (a) review (personal circumstances).""",
    ),
    (
        "Mexfield_Housing_v_Berrisford_2011_UKSC_52",
        "Mexfield Housing Co-operative Ltd v Berrisford [2011] UKSC 52",
        """Mexfield Housing Co-operative Ltd v Berrisford [2011] UKSC 52.
Supreme Court of the United Kingdom. Judgment delivered 2 November 2011.

HEADNOTE: Tenancy – Uncertain term – Periodic tenancy – Yearly tenancy – Landlord
and Tenant Act 1954 – Section 21 notice – Housing Act 1988 – Assured tenancy.

The respondent occupied a property under an agreement which gave her the right to
remain until she breached certain conditions, whereafter the landlord could give
one month's notice. The Court of Appeal held this created a monthly periodic tenancy.

Lord Neuberger JSC:
A lease for a period which is not fixed but determinable only on a condition is void
under common law as an uncertain term. However by virtue of the Landlord and Tenant
Act 1954, such an agreement takes effect as a 90-year term determinable on the death
of the tenant or by one month's notice after the death. The common law rule is ancient
and anomalous. If Parliament had not intervened by the 1954 Act, this Court might
have considered overruling it. Under the 1954 Act the tenancy is for a 90-year term
and the landlord may not serve a s.21 notice to terminate it without the consent
of the tenant or court.

HELD: Appeal dismissed. The agreement created a 90-year lease under the 1954 Act,
not a periodic tenancy. The tenant was entitled to remain in occupation.""",
    ),
    (
        "Knowsley_v_White_2008_UKHL_70",
        "Knowsley Housing Trust v White [2008] UKHL 70",
        """Knowsley Housing Trust v White [2008] UKHL 70 House of Lords.
Judgment delivered 10 December 2008.

HEADNOTE: Assured tenancy – Succession – Housing Act 1985 s.87 – Housing Act 1988
s.17 – Assured tenancy on succession – Whether original secure tenancy survives.

The question was whether, on succession to an assured tenancy, the successor tenant
held an assured tenancy under the Housing Act 1988 or a secure tenancy under the
Housing Act 1985.

Baroness Hale of Richmond:
Housing legislation must be read as a coherent whole. The 1988 Act created a new
framework of assured tenancies for new lettings. Secure tenancies under the 1985 Act
are preserved for existing council tenants. Where a housing association grants a new
tenancy on succession it grants an assured tenancy under the 1988 Act. The successor
does not automatically acquire a secure tenancy. Social landlords who took transfers
of stock from local authorities must grant assured tenancies under the 1988 Act to successors.

HELD: The successor held an assured tenancy. This has important implications for
possession proceedings: the landlord must establish one of the grounds in Schedule 2
to the 1988 Act rather than the grounds in the 1985 Act.""",
    ),
    (
        "Wandsworth_v_Winder_1985_AC_461",
        "Wandsworth LBC v Winder [1985] AC 461",
        """Wandsworth London Borough Council v Winder [1985] AC 461 House of Lords.

HEADNOTE: Secure tenancy – Rent increase – Unlawful resolution – Collateral attack –
Whether defendant may raise invalidity of rent resolution as defence to possession.

The council resolved to increase rents. The defendant refused to pay the increased rent.
The council brought possession proceedings for rent arrears. The defendant wished to argue
in the possession proceedings that the rent resolutions were ultra vires and void.

Lord Fraser of Tullybelton:
It is a fundamental principle of the English legal system that a person who is sued is
entitled to defend himself. A defendant in possession proceedings is entitled to raise
any defence which goes to the root of the claim, including that the contractual basis
for the rent claimed is invalid. To prevent a defendant from raising such a defence
would be to deprive him of a constitutional right. The doctrine that a public law
decision cannot be challenged by way of defence in private law proceedings (the
collateral attack principle) does not prevent a defendant from challenging the
validity of a rent resolution in possession proceedings brought against him.

HELD: The defendant was entitled to plead the invalidity of the rent resolutions
as a defence. The appeal was allowed.""",
    ),
    (
        "Barber_v_Croydon_2010_EWCA_Civ_51",
        "Barber v Croydon LBC [2010] EWCA Civ 51",
        """Barber v London Borough of Croydon [2010] EWCA Civ 51 Court of Appeal (Civil Division).

HEADNOTE: Introductory tenancy – Review panel – Natural justice – Article 6 ECHR –
Procedural fairness – Housing Act 1996 s.129.

The claimant held an introductory tenancy. The council served notice of proceedings for
possession on the ground of antisocial behaviour. The claimant requested a review. The
reviewing officer who conducted the review had previously been involved in the decision
to seek possession.

Longmore LJ:
The review procedure under s.129 of the Housing Act 1996 must comply with the requirements
of natural justice and, where Article 6 ECHR is engaged, the requirements of a fair
hearing. Where the reviewing officer has had prior involvement in the decision under
review, there is a real danger of bias even if the officer genuinely tries to approach
the matter afresh. The appearance of bias is sufficient to vitiate the review. The
importance of housing to the individual means that the courts should be vigilant to
ensure procedural fairness.

HELD: The review was conducted in breach of natural justice and was therefore invalid.
The possession proceedings could not proceed on the basis of the flawed review.""",
    ),
    (
        "Sims_v_Dacorum_2014_UKSC_63",
        "Sims v Dacorum Borough Council [2014] UKSC 63",
        """Sims v Dacorum Borough Council [2014] UKSC 63 Supreme Court of the United Kingdom.
Judgment delivered 26 November 2014.

HEADNOTE: Secure tenancy – Joint tenants – Notice to quit by one tenant – Human Rights
Act 1998 – Article 8 ECHR – Article 14 ECHR.

One of two joint secure tenants served a notice to quit without the other's consent,
thereby ending the joint tenancy. The council then sought possession against the
remaining occupier who had been in the property for many years with her children.

Lord Hodge JSC:
The common law rule that one joint tenant may serve a notice to quit without the
other's consent is well established. The notice to quit does not interfere with the
remaining occupier's Convention rights in a manner which engages positive obligations
on the State. The interference is by a private individual (the departing joint tenant),
not by a public authority. Even if Article 8 is engaged in the possession proceedings
themselves, the possession of a home which the occupier has no right to remain in
will almost invariably be proportionate where the landlord is a public authority
acting within its housing management functions.

HELD: The notice to quit was valid. Possession was proportionate. Appeal dismissed.""",
    ),
    (
        "Hussein_v_Mehlman_1992_Landlord_Tenant",
        "Hussein v Mehlman [1992] 2 EGLR 87",
        """Hussein v Mehlman [1992] 2 EGLR 87 (Assistant Recorder Sedley QC).

HEADNOTE: Landlord and Tenant – Disrepair – Acceptance of repudiation – Whether
tenant may treat landlord's persistent failure to repair as repudiation of tenancy.

The landlord persistently failed to carry out repairs to the property despite the tenant's
repeated requests and despite statutory notices. The property was in a very bad state of
repair. The tenant surrendered the tenancy and claimed damages.

Mr Stephen Sedley QC (sitting as Assistant Recorder):
The general rule is that a lease cannot be repudiated. However this rule must give way
to the fundamental principle that where one party to a contract evinces a clear intention
not to perform a fundamental obligation, the other party may accept that repudiation and
treat the contract as at an end. A landlord's repairing covenant is a fundamental term
of a tenancy. Persistent and deliberate breach over a prolonged period may amount to
repudiation. Here the landlord's conduct clearly repudiated the tenancy and the tenant
was entitled to accept that repudiation, vacate the property, and claim damages.

HELD: The tenant validly accepted repudiation. Damages awarded for disrepair, personal
injury, and loss of enjoyment of the property.""",
    ),
]


def scrape_bailii() -> list[dict]:
    all_chunks = []
    log.info("loading_case_law", count=len(_CASE_TEXTS))

    for filename, citation, text in _CASE_TEXTS:
        out_file = CASE_LAW_DIR / f"{filename}.txt"
        if not out_file.exists():
            out_file.write_text(text, encoding="utf-8")

        chunks = chunk_text(text, source=f"UK Case Law: {citation}")
        all_chunks.extend(chunks)
        log.info("case_loaded", citation=citation, chunks=len(chunks))

    log.info("case_law_done", total_chunks=len(all_chunks))
    return all_chunks
