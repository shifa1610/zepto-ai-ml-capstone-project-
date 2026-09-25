## 1. Books over GBP 30, ordered by price, limited to 10

```sql
SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp > 30
        ORDER BY price_gbp DESC
        LIMIT 10
```

                                                                                                                          title  price_gbp  rating
                                                                                   The Death of Humanity: and the Case for Life      58.11       4
                                                                                                 Slow States of Collapse: Poems      57.31       3
                                             Our Band Could Be Your Life: Scenes from the American Indie Underground, 1981-1991      57.25       3
                                                                                                            The Past Never Ends      56.50       4
The Pioneer Woman Cooks: Dinnertime: Comfort Classics, Freezer Food, 16-Minute Meals, and Other Delicious Ways to Solve Supper!      56.41       1
                                                                                                              Masks and Shadows      56.40       2
                                                                                                The Secret of Dreadwillow Carse      56.13       1
                                                                 The Electric Pencil: Drawings from Inside State Hospital No. 3      56.06       1
                                                                                                  Birdsong: A Story in Pictures      54.64       3
                                                                                          Sapiens: A Brief History of Humankind      54.23       5

## 2. Distinct categories

```sql
SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
```

     category_name
     Add a comment
               Art
          Business
         Childrens
      Contemporary
           Default
           Fantasy
           Fiction
    Food and Drink
            Health
Historical Fiction
           History
            Horror
             Music
           Mystery
         New Adult
        Nonfiction
        Philosophy
            Poetry
          Politics
           Romance
           Science
   Science Fiction
         Self Help
    Sequential Art
      Spirituality
          Thriller
            Travel
       Young Adult

## 3. Books rated 4 or 5 (IN)

```sql
SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 10
```

                                                                            title  rating  price_gbp
               #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.       5      23.11
                                                                       Black Dust       5      34.53
                                                       Chase Me (Paris Nights #2)       5      25.27
                                                                             Join       5      35.67
                                 Princess Between Worlds (Wide-Awake Princess #5)       5      13.34
Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1)       5      13.61
                                                      Private Paris (Private #10)       5      47.61
                                                        Rip it Up and Start Again       5      35.02
                                            Sapiens: A Brief History of Humankind       5      54.23
                          Scott Pilgrim's Precious Little Life (Scott Pilgrim #1)       5      52.29

## 4. Books priced between GBP 20 and GBP 40 (BETWEEN)

```sql
SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
        LIMIT 10
```

                                                                                                                                                 title  price_gbp
                                                                    The Inefficiency Assassin: Time Management Tactics for Working Smarter, Not Longer      20.59
                                                                                                                                 Shakespeare's Sonnets      20.66
                                                                                                             In the Country We Love: My Family Divided      22.00
                                           America's Cradle of Quarterbacks: Western Pennsylvania's Football Factory from Johnny Unitas to Joe Montana      22.50
                                                        The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics      22.60
                                                                                                                                       The Requiem Red      22.65
                                                                                    #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.      23.11
                                                                                                                                     The Elephant Tree      23.82
                                                                                                                                                  Olio      23.88
The Mindfulness and Acceptance Workbook for Anxiety: A Guide to Breaking Free from Anxiety, Phobias, and Worry Using Acceptance and Commitment Therapy      23.89

## 5. JOIN query

```sql
SELECT c.category_name, b.title, b.rating, b.price_gbp
    FROM books AS b
    JOIN categories AS c ON b.category_id = c.category_id
    ORDER BY b.rating DESC, b.price_gbp ASC, b.title ASC
    LIMIT 10
```

### SQLite result

 category_name                                                                              title  rating  price_gbp
       Fantasy                                   Princess Between Worlds (Wide-Awake Princess #5)       5      13.34
Sequential Art  Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1)       5      13.61
    Philosophy                                                                     Sophie's World       5      15.94
       Fiction                                                                             Thirst       5      17.27
   Young Adult                                                                        Set Me Free       5      17.46
  Spirituality                         The Four Agreements: A Practical Guide to Personal Freedom       5      17.66
       Default The Inefficiency Assassin: Time Management Tactics for Working Smarter, Not Longer       5      20.59
    Nonfiction                 #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.       5      23.11
      Thriller                                                                  The Elephant Tree       5      23.82
       Romance                                                         Chase Me (Paris Nights #2)       5      25.27

### pandas merge result

 category_name                                                                              title  rating  price_gbp
       Fantasy                                   Princess Between Worlds (Wide-Awake Princess #5)       5      13.34
Sequential Art  Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1)       5      13.61
    Philosophy                                                                     Sophie's World       5      15.94
       Fiction                                                                             Thirst       5      17.27
   Young Adult                                                                        Set Me Free       5      17.46
  Spirituality                         The Four Agreements: A Practical Guide to Personal Freedom       5      17.66
       Default The Inefficiency Assassin: Time Management Tactics for Working Smarter, Not Longer       5      20.59
    Nonfiction                 #HigherSelfie: Wake Up Your Life. Free Your Soul. Find Your Tribe.       5      23.11
      Thriller                                                                  The Elephant Tree       5      23.82
       Romance                                                         Chase Me (Paris Nights #2)       5      25.27

**Results match:** True
