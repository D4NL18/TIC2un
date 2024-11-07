import { TestBed } from '@angular/core/testing';

import { NeurofuzzyService } from './neurofuzzy.service';

describe('NeurofuzzyService', () => {
  let service: NeurofuzzyService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(NeurofuzzyService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
