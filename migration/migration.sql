create extension if not exists vector;
-- Create documents table
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(1536)
);
-- Create similarity search function
create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  where (filter = '{}' or d.metadata @> filter)
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
-- Create index for faster searches
create index if not exists documents_embedding_idx 
  on documents 
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);