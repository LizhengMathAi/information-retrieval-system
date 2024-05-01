import time
import sqlite3
import openreview


# API V2
CLIENT = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username='zli66@nyit.edu', password='Date@286053')
DB_NAME = 'src/papers.sqlite'


def display_venue():
    # List all venues IDs.
    venues = CLIENT.get_group(id='venues').members
    # for venue_id in venues:
    #     if "NeurIPS" in venue_id:
    #         print(venue_id)
    return venues


def load_venue(venue_id, accepted_only=True):
    if accepted_only:
        # To only get "accepted" submissions, you'll need to query the notes by venueid.
        submissions = CLIENT.get_all_notes(content={'venueid': venue_id} )
    else:
        venue_group = CLIENT.get_group(venue_id)
        submission_name = venue_group.content['submission_name']['value']
        submissions = CLIENT.get_all_notes(invitation=f'{venue_id}/-/{submission_name}')
    print(submissions)

    # Connect to SQLite database
    conn = sqlite3.connect(database=DB_NAME)

    # Create tables
    conn.execute('''CREATE TABLE IF NOT EXISTS Papers (
                        id TEXT PRIMARY KEY, 
                        cdate INTEGER, 
                        ddate INTEGER,
                        mdate INTEGER, 
                        odate INTEGER, 
                        pdate INTEGER, 
                        tcdate INTEGER, 
                        tmdate INTEGER, 
                        domain TEXT, 
                        forum TEXT, 
                        number INTEGER
                    );''')

    conn.execute('''CREATE TABLE IF NOT EXISTS Content (
                        paper_id TEXT,
                        title TEXT, 
                        authorids TEXT, 
                        authors TEXT, 
                        keywords TEXT, 
                        abstract TEXT, 
                        venue TEXT, 
                        venueid TEXT,
                        pdf TEXT,
                        FOREIGN KEY(paper_id) REFERENCES Papers(id)
                    );''')

    # Insert data into the tables (Example for one paper object)
    for paper in submissions: # your paper object here
        # Insert into Papers table
        conn.execute('''INSERT INTO Papers (id, cdate, ddate, mdate, odate, pdate, tcdate, tmdate, domain, forum, number)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (paper.id, paper.cdate, paper.ddate, paper.mdate,  paper.odate, paper.pdate, paper.tcdate, paper.tmdate, 
                    paper.domain, paper.forum, paper.number))

        # Insert into Content table
        content = paper.content
        conn.execute('''INSERT INTO Content (paper_id, title, authorids, authors, keywords, abstract, venue, venueid, pdf)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (paper.id, 
                    content['title']['value'], 
                    '\t'.join(content['authorids']['value']) if 'authorids' in content.keys() else 'NULL', 
                    '\t'.join(content['authors']['value']) if 'author' in content.keys() else 'NULL',
                    '\t'.join(content['keywords']['value']) if 'keywords' in content.keys() else 'NULL',
                    content['abstract']['value'] if 'abstract' in content.keys() else 'NULL', 
                    content['venue']['value'], 
                    content['venueid']['value'], 
                    content['pdf']['value'] if 'pdf' in content.keys() else 'NULL', 
                    ))

    # Commit and close
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # display_venue()
    # for year in range(2013, 2024):
    #     load_venue(venue_id='ICLR.cc/{}/Conference'.format(year))
    # for year in range(2013, 2025):
    #     load_venue(venue_id='ICLR.cc/{}/Conference'.format(year))
    # for year in range(2020, 2024):
    #     load_venue(venue_id='NeurIPS.cc/{}/Conference'.format(year))

    venues = display_venue()
    for venue_id in venues:
        print(venue_id)
        load_venue(venue_id)
