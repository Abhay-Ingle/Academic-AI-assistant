# DBMS - Normalization and Transactions

Normalization organises a relational schema to reduce redundancy and avoid update, insert and delete anomalies.

## Normal forms
- 1NF: every attribute holds atomic (indivisible) values and there are no repeating groups.
- 2NF: the table is in 1NF and no non-key attribute depends on only part of a composite primary key (no partial dependency).
- 3NF: the table is in 2NF and no non-key attribute depends on another non-key attribute (no transitive dependency).
- BCNF: for every functional dependency X -> Y, X must be a super key.

## Transactions and ACID
A transaction is a logical unit of work. ACID properties: Atomicity (all or nothing), Consistency (database moves from one valid state to another), Isolation (concurrent transactions do not interfere) and Durability (committed changes survive failures).

## Indexing
An index is a data structure, commonly a B+ tree, that speeds up searches on a column at the cost of extra storage and slower writes. A clustered index determines the physical order of rows; a table can have only one.
