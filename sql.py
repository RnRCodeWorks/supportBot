import sqlparse
from sqlparse.tokens import Token

sql = 'SELECT  @FIELD  FROM dbo.tblInItem i LEFT JOIN dbo.tblInItemLoc l ON i.ItemId = l.ItemId LEFT JOIN dbo.tblInHazMat h ON i.HMRef = h.HMRef LEFT JOIN (SELECT  ItemId, LocId, COUNT(*) AS QtyOnHand FROM  dbo.tblInItemSer WHERE SerNumStatus = 1 GROUP BY ItemId, LocId, LotNum UNION ALL SELECT ItemId, LocId, SUM(Qty - InvoicedQty - RemoveQty) AS QtyOnHand FROM dbo.tblInQtyOnHand GROUP BY ItemId, LocId) o ON i.ItemId = o.ItemId AND l.LocId = o.LocId LEFT JOIN (SELECT  ItemId, LocId, SUM(CASE WHEN TransType = 0 THEN Qty ELSE 0 END) AS QtyCmtd, SUM(CASE WHEN TransType = 2 THEN Qty ELSE 0 END) AS QtyOnOrder FROM dbo.tblInQty GROUP BY ItemId, LocId) t ON i.ItemId = t.ItemId AND l.LocId = t.LocId AND @DATE >= "@BEGINDATE" AND @DATE <= "@ENDDATE"'
parsed = sqlparse.parse(sql, encoding='Oracle')[0]
# for x in parsed.tokens:
# 	if x.is_group:
# 		for t in x.tokens:
# 			if t.ttype == Token.Name:
# 				print(t.value, t.ttype)
# 	else:
# 		if x.ttype == Token.Name:
# 			print(x.value, x.ttype)

lookFor = [Token.Name, Token.Punctuation]
ignore = ['(', ')', ',', 'SUM']
tables = set([])
# precededByPunctuation = False


def parsetokens(tokens):
	for x in tokens:
		if x.is_group:
			print('group: ', x)
			parsetokens(x)
		else:
			if x.ttype in lookFor and x.value not in ignore:
				print(x.value, x.ttype)


parsetokens(parsed.tokens)
