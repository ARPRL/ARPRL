# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:50:08 2022

@author: SirLagrange
"""

from cox import store

pri, robust, cla = 'income_n_pri', 'income_n_robust', 'income_n_cla'
print(pri, robust, cla)
s = store.Store('D:/path/to/output', pri)
tb1 = s['logs'].df
privacy_acc = tb1['pri_prec1']
s = store.Store('D:/path/to/output', robust)
tb2 = s['logs'].df
adv_acc = tb2['adv_prec1']
nat_acc = tb2['nat_prec1']
s = store.Store('D:/path/to/output', cla)
tb3 = s['logs'].df
cla_acc = tb3['nat_prec1']
print('privacy_acc {privacy_acc:.2f} | '
      'adv_acc {adv_acc:.2f} | '
      'nat_acc {nat_acc:.2f} | '
      'cla_acc {cla_acc:.2f} | '
      .format(privacy_acc=privacy_acc[len(privacy_acc) - 1],
              adv_acc=adv_acc[len(adv_acc) - 1],
              nat_acc=nat_acc[len(nat_acc) - 1],
              cla_acc=cla_acc[len(cla_acc) - 1]))
s.close()
