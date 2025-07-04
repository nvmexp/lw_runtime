#include "stack.h"
#include <string.h>

int
sd_stack_grow(struct sd_stack *st, size_t new_size)
{
	void **new_st;

	if (st->asize >= new_size)
		return 0;

	new_st = realloc(st->item, new_size * sizeof(void *));
	if (new_st == NULL)
		return -1;

	memset(new_st + st->asize, 0x0,
		(new_size - st->asize) * sizeof(void *));

	st->item = new_st;
	st->asize = new_size;

	if (st->size > new_size)
		st->size = new_size;

	return 0;
}

void
sd_stack_free(struct sd_stack *st)
{
	if (!st)
		return;

	free(st->item);

	st->item = NULL;
	st->size = 0;
	st->asize = 0;
}

int
sd_stack_init(struct sd_stack *st, size_t initial_size)
{
	st->item = NULL;
	st->size = 0;
	st->asize = 0;

	if (!initial_size)
		initial_size = 8;

	return sd_stack_grow(st, initial_size);
}

void *
sd_stack_pop(struct sd_stack *st)
{
	if (!st->size)
		return NULL;

	return st->item[--st->size];
}

int
sd_stack_push(struct sd_stack *st, void *item)
{
	if (sd_stack_grow(st, st->size * 2) < 0)
		return -1;

	st->item[st->size++] = item;
	return 0;
}

void *
sd_stack_top(struct sd_stack *st)
{
	if (!st->size)
		return NULL;

	return st->item[st->size - 1];
}

